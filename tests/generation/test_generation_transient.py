# ruff: noqa: S101, PLR2004, SLF001
"""Physical-unit transient one-hour index and lazy-loader contracts."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
import pytest
import torch

from src import common, datasets, generation, learning
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _json(value: object) -> str:
    """Return compact HDF5 JSON metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _synthetic_scientific_contract() -> dict[str, Any]:
    """Return the explicit config evidence owned by one synthetic case."""
    return {
        "schema_version": generation.cases.config.CONFIG_SCHEMA_VERSION,
        "simulation_profile": "transient_drying",
        "grid": {
            "nx": 401,
            "ny": 251,
            "Lx": 1.2,
            "Ly": 0.75,
            "Lz": 0.8,
            "boundaries_included": True,
            "dx": 0.003,
            "dy": 0.003,
        },
        "time": {
            "start": 0.0,
            "stop": 168.0,
            "interval": 1.0,
            "internal_steps": "adaptive",
            "irregular_stop_state": "diagnostic_only",
            "regular_times": [float(index) for index in range(169)],
        },
        "storage": {
            "schema_version": generation.publication.storage.HDF5_SCHEMA_VERSION,
            "converter_version": generation.publication.storage.HDF5_CONVERTER_VERSION,
            "compression": "gzip",
            "compression_level": 4,
            "shuffle": True,
            "float32_rtol": 1e-6,
            "float32_atol": 1e-7,
            "chunk_time": 1,
            "chunk_y": 64,
            "chunk_x": 64,
        },
        "scientific_fixed_values": {
            "T_flow_ref": 300.65,
            "p_ref": 101325.0,
            "p_out": 0.0,
            "f_wet_dm_max": 0.05,
        },
    }


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
    case_inputs, case_outputs = datasets.packages.generated_batch._steady_flow_fields(
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
    fingerprint = datasets.contracts.identity.compute_case_fingerprint(
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
    exact_stop_time: float | None = 2.5,
    case_input_id: str = "1" * 64,
    simulation_case_id: str = "2" * 64,
    scientific_contract: dict[str, Any] | None = None,
    regular_state_count: int = 3,
) -> np.ndarray:
    """Write one compressed canonical case with an optional exact-stop state."""
    scientific = _synthetic_scientific_contract() if scientific_contract is None else scientific_contract
    grid = scientific["grid"]
    storage = scientific["storage"]
    x_axis = np.linspace(0.0, float(grid["Lx"]), int(grid["nx"]), dtype=np.float64)
    y_axis = np.linspace(0.0, float(grid["Ly"]), int(grid["ny"]), dtype=np.float64)
    shape = (y_axis.size, x_axis.size)
    static_constants = (1.0e-10, 0.0, 2.0e-10, 0.4, 100.0, 0.2, 0.1, 0.05, 101325.0, 550.0)
    assert len(static_constants) == len(profiles.TRANSIENT_STATIC_FIELD_NAMES)
    static = np.stack([np.full(shape, value, dtype=np.float32) for value in static_constants])
    regular_time = np.asarray(scientific["time"]["regular_times"][:regular_state_count], dtype=np.float64)
    initial_water = static_constants[-1] * static_constants[5]
    base = np.asarray([295.0, 0.4, initial_water, initial_water], dtype=np.float32)
    increments = np.asarray([1.0, 0.01, -0.5, -0.25], dtype=np.float32)

    def state_at(time_value: float) -> np.ndarray:
        return np.stack([np.full(shape, base[channel] + time_value * increments[channel], dtype=np.float32) for channel in range(4)])

    transient = np.stack([state_at(float(time_value)) for time_value in regular_time])
    exact_fields = None if exact_stop_time is None else state_at(exact_stop_time)
    complete_time = regular_time if exact_stop_time is None else np.concatenate((regular_time, [exact_stop_time]))
    complete_fields = transient if exact_fields is None else np.concatenate((transient, exact_fields[np.newaxis, ...]))
    scalar_values = {
        "T_amb": 295.0,
        "eps_bed_cal_ref": 0.5,
        "rho_bu_dry_ref": 550.0,
        "k_gr": 0.2,
        "cp_gr_dry": 1300.0,
        "X_target_wb": 0.12,
        "r_surf_0": 2.0e-5,
        "r_int_surf": 1.5,
        "f_surf": 0.4,
        "A_osw": 0.1,
        "B_osw": 0.002,
        "C_osw": 0.3,
    }
    scalar_names = profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    scalars = np.asarray([scalar_values[name] for name in scalar_names], dtype=np.float64)
    ownership = ["case_dependent"] * len(scalar_names)
    schedule_times = np.asarray(scientific["time"]["regular_times"], dtype=np.float64)
    schedule = np.zeros((schedule_times.size, len(profiles.SCHEDULE_FIELDS)), dtype=np.float64)
    schedule[:, 0] = schedule_times
    schedule[:, 1] = 295.0 + 0.01 * schedule[:, 0]
    schedule[:, 2] = 0.009
    schedule[:, 3] = schedule_service.humidity_ratio_to_relative_humidity(
        schedule[:, 2],
        schedule[:, 1],
        pressure=float(scientific["scientific_fixed_values"]["p_ref"]),
    )
    global_values = np.zeros((complete_time.size, len(profiles.GLOBAL_FIELD_NAMES)), dtype=np.float64)
    global_values[:, 0] = complete_time
    f_surf = scalar_values["f_surf"]
    rho = static_constants[-1]
    for index, fields in enumerate(complete_fields):
        water = f_surf * float(fields[2, 0, 0]) + (1.0 - f_surf) * float(fields[3, 0, 0])
        x_wb = water / (rho + water)
        global_values[index] = [
            complete_time[index],
            x_wb,
            1.0 if index == 0 else 0.04,
            float(grid["Lz"]) * float(grid["Lx"]) * float(grid["Ly"]) * water,
            1.0,
            0.1,
            0.02,
            0.01,
            0.0,
            float(fields[0, 0, 0]),
            float(fields[1, 0, 0]),
        ]
    final_fields = complete_fields[-1]
    final_water = f_surf * float(final_fields[2, 0, 0]) + (1.0 - f_surf) * float(final_fields[3, 0, 0])
    final_x_wb = final_water / (rho + final_water)
    final_status = np.asarray(
        [
            complete_time[-1],
            global_values[-1, profiles.GLOBAL_FIELD_NAMES.index("f_wet_dm")],
            final_x_wb,
            final_x_wb,
            float(final_fields[0, 0, 0]),
            float(final_fields[0, 0, 0]),
            float(final_fields[1, 0, 0]),
            float(final_fields[1, 0, 0]),
        ],
        dtype=np.float64,
    )
    profile = profiles.get_profile("transient_drying")
    scientific_digest = common.serialization.canonical_json_sha256(scientific)
    scalar_entries = [
        {"name": name, "value": scalar_values[name], "unit": unit, "owner": owner}
        for name, unit, owner in zip(
            scalar_names,
            profiles.TRANSIENT_SCALAR_INPUT_UNITS,
            ownership,
            strict=True,
        )
    ]
    scalar_handoff = {
        "mechanism": "case_local_long_form_csv",
        "filename": "scalars.csv",
        "fresh_per_case": True,
        "runtime_validation": "required",
        "entries": scalar_entries,
    }
    conversion_pressure = {"name": "p_ref", "value": 101325.0, "unit": "Pa", "owner": "package_fixed"}
    case_scientific_provenance = {
        "schema_kind": "vp2_case_scientific_provenance",
        "schema_version": 1,
        "case_id": "case_0001",
        "case_index": 1,
        "case_input_id": case_input_id,
        "simulation_case_id": simulation_case_id,
        "material_family": "lentil",
        "material_role": "seen",
        "evaluation_regime": "id",
        "sampling_regime": "natural",
        "natural_support_state": "natural",
        "seed_evidence": {},
        "block_provenance": {"airflow": {}, "initial_moisture": {}, "operation": {}, "material_properties": {}},
        "conditional_supports": {},
        "sampled_values": {name: scalar_values[name] for name in scalar_names},
        "sampled_units": dict(zip(scalar_names, profiles.TRANSIENT_SCALAR_INPUT_UNITS, strict=True)),
        "coupled_selections": {},
        "ood": {"natural_support_state": "natural"},
        "spatial_diagnostics": {},
        "schedule_diagnostics": {},
    }

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_kind": generation.publication.storage.HDF5_SCHEMA_KIND,
                "schema_version": generation.publication.storage.HDF5_SCHEMA_VERSION,
                "converter_version": generation.publication.storage.HDF5_CONVERTER_VERSION,
                "simulation_profile": "transient_drying",
                "case_id": "case_0001",
                "case_index": 1,
                "material_family": "lentil",
                "material_role": "seen",
                "evaluation_regime": "id",
                "sampling_regime": "natural",
                "natural_support_state": "natural",
                "case_input_id": case_input_id,
                "simulation_case_id": simulation_case_id,
                "scientific_config_digest": scientific_digest,
                "export_contract_sha256": "4" * 64,
                "airflow_source": "comsol_coupled_reference",
                "git_commit": "a" * 40,
                "template_relative_path": profile.template_relative_path,
                "template_sha256": profile.template_sha256,
                "available_learning_views": _json(["steady_flow", "transient_drying"]),
            }
        )
        provenance = handle.create_group("provenance")
        provenance_values = {
            "scientific_config_json": scientific,
            "case_scientific_provenance_json": case_scientific_provenance,
            "input_files_json": {
                "fields.csv": {"sha256": "6" * 64, "size_bytes": 1},
                "scalars.csv": {"sha256": "b" * 64, "size_bytes": 1},
                "schedule.csv": {"sha256": "c" * 64, "size_bytes": 1},
            },
            "source_exports_json": {
                "exports/steady.csv": {"role": "steady_flow_fields", "sha256": "7" * 64, "size_bytes": 1},
                "exports/transient.csv": {"role": "transient_fields", "sha256": "8" * 64, "size_bytes": 1},
                "exports/global.csv": {"role": "global_time_series", "sha256": "9" * 64, "size_bytes": 1},
                "exports/final.csv": {"role": "final_status", "sha256": "a" * 64, "size_bytes": 1},
            },
            "template_json": {
                "relative_path": profile.template_relative_path,
                "filename": profile.template_path.name,
                "sha256": profile.template_sha256,
                "sha256_validation": "pass",
                "comsol_internal_contract": "runtime_validation_required",
            },
            "scalar_handoff_json": scalar_handoff,
            "stationary_fixed_ownership_json": {
                "T_flow_ref": {"owner": "package_fixed", "unit": "K", "fixed_value": 300.65},
                "p_ref": {"owner": "package_fixed", "unit": "Pa", "fixed_value": 101325.0},
                "p_out": {"owner": "package_fixed", "unit": "Pa", "fixed_value": 0.0},
            },
        }
        for name, value in provenance_values.items():
            provenance.create_dataset(name, data=_json(value), dtype=h5py.string_dtype(encoding="utf-8"))
        coordinates = handle.create_group("coords")
        x_dataset = coordinates.create_dataset("x", data=x_axis)
        y_dataset = coordinates.create_dataset("y", data=y_axis)
        x_dataset.attrs["unit"] = "m"
        y_dataset.attrs["unit"] = "m"
        static_dataset = handle.create_group("static").create_dataset(
            "fields",
            data=static,
            chunks=(
                1,
                min(int(storage["chunk_y"]), y_axis.size),
                min(int(storage["chunk_x"]), x_axis.size),
            ),
            compression=str(storage["compression"]),
            compression_opts=int(storage["compression_level"]),
            shuffle=bool(storage["shuffle"]),
        )
        static_dataset.attrs["field_names"] = _json(list(profiles.TRANSIENT_STATIC_FIELD_NAMES))
        static_dataset.attrs["units"] = _json(list(profiles.TRANSIENT_STATIC_FIELD_UNITS))
        scalar_dataset = handle.create_group("scalar").create_dataset("values", data=scalars)
        scalar_dataset.attrs["field_names"] = _json(list(scalar_names))
        scalar_dataset.attrs["units"] = _json(list(profiles.TRANSIENT_SCALAR_INPUT_UNITS))
        scalar_dataset.attrs["ownership"] = _json(ownership)
        time_dataset = handle.create_dataset("time", data=regular_time)
        time_dataset.attrs["unit"] = "h"
        time_dataset.attrs["classification_atol"] = generation.publication.storage.time_classification_tolerance(scientific["time"])
        time_dataset.attrs["classification_basis"] = generation.publication.storage._time_classification_basis(scientific["time"])
        transient_dataset = handle.create_group("transient").create_dataset(
            "fields",
            data=transient,
            chunks=(
                min(int(storage["chunk_time"]), regular_time.size),
                1,
                min(int(storage["chunk_y"]), y_axis.size),
                min(int(storage["chunk_x"]), x_axis.size),
            ),
            compression=str(storage["compression"]),
            compression_opts=int(storage["compression_level"]),
            shuffle=bool(storage["shuffle"]),
        )
        transient_dataset.attrs["field_names"] = _json(list(profiles.TRANSIENT_FIELD_NAMES))
        transient_dataset.attrs["units"] = _json(list(profiles.TRANSIENT_FIELD_UNITS))
        if exact_fields is not None:
            exact_group = handle.create_group("exact_stop")
            exact_time_dataset = exact_group.create_dataset("time", data=np.asarray([exact_stop_time], dtype=np.float64))
            exact_time_dataset.attrs["unit"] = "h"
            exact_dataset = exact_group.create_dataset(
                "fields",
                data=exact_fields,
                chunks=(
                    1,
                    min(int(storage["chunk_y"]), y_axis.size),
                    min(int(storage["chunk_x"]), x_axis.size),
                ),
                compression=str(storage["compression"]),
                compression_opts=int(storage["compression_level"]),
                shuffle=bool(storage["shuffle"]),
            )
            exact_dataset.attrs["field_names"] = _json(list(profiles.TRANSIENT_FIELD_NAMES))
            exact_dataset.attrs["units"] = _json(list(profiles.TRANSIENT_FIELD_UNITS))
            exact_group.attrs["usage"] = "diagnostic_only_no_training_transition"
        schedule_dataset = handle.create_group("schedule").create_dataset(
            "values",
            data=schedule,
            compression=str(storage["compression"]),
            compression_opts=int(storage["compression_level"]),
            shuffle=bool(storage["shuffle"]),
        )
        schedule_dataset.attrs["field_names"] = _json(list(profiles.SCHEDULE_FIELDS))
        schedule_dataset.attrs["units"] = _json(list(profiles.SCHEDULE_UNITS))
        schedule_dataset.attrs["conversion_pressure"] = _json(conversion_pressure)
        schedule_dataset.attrs["humidity_conversion_owner"] = "generation_schedule"
        global_dataset = handle.create_group("global").create_dataset(
            "values",
            data=global_values,
            compression=str(storage["compression"]),
            compression_opts=int(storage["compression_level"]),
            shuffle=bool(storage["shuffle"]),
        )
        global_dataset.attrs["field_names"] = _json(list(profiles.GLOBAL_FIELD_NAMES))
        global_dataset.attrs["units"] = _json(list(profiles.GLOBAL_FIELD_UNITS))
        final_dataset = handle.create_group("final_status").create_dataset("values", data=final_status)
        final_dataset.attrs["field_names"] = _json(list(profiles.FINAL_STATUS_FIELDS))
        final_dataset.attrs["units"] = _json(list(profiles.FINAL_STATUS_UNITS))
    return increments


def test_hdf5_validation_uses_embedded_noncurrent_contract(tmp_path: Path) -> None:
    """Prove grid, time, compression, and chunks are resolved rather than frozen."""
    scientific = _synthetic_scientific_contract()
    scientific["grid"].update(
        {
            "nx": 9,
            "ny": 7,
            "Lx": 0.8,
            "Ly": 0.6,
            "Lz": 0.4,
            "dx": 0.1,
            "dy": 0.1,
        }
    )
    scientific["time"].update(
        {
            "stop": 2.0,
            "interval": 0.5,
            "regular_times": [0.0, 0.5, 1.0, 1.5, 2.0],
        }
    )
    scientific["storage"].update(
        {
            "compression_level": 6,
            "chunk_time": 2,
            "chunk_y": 3,
            "chunk_x": 4,
        }
    )
    source = tmp_path / "noncurrent.h5"
    _write_transient_case(
        source,
        exact_stop_time=1.25,
        scientific_contract=scientific,
    )

    admitted = generation.publication.storage.validate_case_hdf5(
        source,
        expected_profile="transient_drying",
    )
    assert admitted["scientific_config_digest"] == common.serialization.canonical_json_sha256(scientific)
    with h5py.File(source, "r") as handle:
        assert _hdf5_dataset(handle, "coords/x").shape == (9,)
        assert _hdf5_dataset(handle, "coords/y").shape == (7,)
        assert _hdf5_dataset(handle, "static/fields").chunks == (1, 3, 4)
        transient = _hdf5_dataset(handle, "transient/fields")
        assert transient.shape == (3, 4, 7, 9)
        assert transient.chunks == (2, 1, 3, 4)
        assert transient.compression == "gzip"
        assert transient.compression_opts == 6
        assert _hdf5_dataset(handle, "schedule/values").shape[0] == 5


def _source(
    path: Path,
    *,
    regime: str = "id",
    membership: str = "train",
    package_case_id: str = "synthetic_transient__case_0001",
    case_input_id: str = "1" * 64,
    simulation_case_id: str = "2" * 64,
) -> datasets.packages.trajectory.TransientSourceCase:
    """Return one typed source bound to the synthetic HDF5 identities."""
    return datasets.packages.trajectory.TransientSourceCase(
        path=path,
        package_case_id=package_case_id,
        source_batch_id="synthetic_transient_batch",
        membership=membership,
        evaluation_regime=regime,
        expected_sha256=common.serialization.file_sha256(path),
        expected_case_input_id=case_input_id,
        expected_simulation_case_id=simulation_case_id,
        material_family="lentil",
        ood_group=None if regime == "id" else "bed",
        ood_parameters=() if regime == "id" else ("kappa_mean",),
        ood_evidence={},
    )


def _build_index(source: Path, index_path: Path, *, regime: str = "id", membership: str = "train") -> dict[str, Any]:
    """Build one compact synthetic transition index."""
    return datasets.packages.trajectory.build_transient_index(
        [_source(source, regime=regime, membership=membership)],
        index_path,
        dataset_name=f"transient_drying__lentil__{regime}",
        dataset_id=f"transient_drying__lentil__{regime}__synthetic",
        evaluation_regime=regime,
        source_root=source.parent,
    )


def _one_step_sampling() -> datasets.contracts.transient.TransientSamplingSpec:
    """Return the explicit one-step runtime sampling choice."""
    return datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition")


def test_transient_index_selects_one_hour_steps_from_finer_regular_source(tmp_path: Path) -> None:
    """Select the learned one-hour transition from a compatible finer source grid."""
    scientific = _synthetic_scientific_contract()
    scientific["time"].update(
        {
            "stop": 2.0,
            "interval": 0.5,
            "regular_times": [0.0, 0.5, 1.0, 1.5, 2.0],
        }
    )
    source = tmp_path / "finer.h5"
    increments = _write_transient_case(
        source,
        exact_stop_time=1.25,
        scientific_contract=scientific,
    )
    payload = _build_index(source, tmp_path / "finer-index.json")
    assert payload["sample_count"] == 1
    assert [(sample["time_index_n"], sample["time_index_n_plus_1"]) for sample in payload["samples"]] == [(0, 2)]

    dataset = datasets.runtime.transient.TransientPhysicalDataset(
        tmp_path / "finer-index.json",
        sampling=_one_step_sampling(),
        source_root=tmp_path,
    )
    item = dataset[0]
    torch.testing.assert_close(
        item["target"],
        torch.from_numpy(np.broadcast_to(increments[:, None, None], item["target"].shape).copy()),
    )


def test_transient_index_rejects_rehashed_noncanonical_contract_digest(tmp_path: Path) -> None:
    """Reject a coordinated index rewrite around an arbitrary contract digest."""
    source = tmp_path / "case.h5"
    _write_transient_case(source)
    index_path = tmp_path / "index.json"
    payload = _build_index(source, index_path)
    payload["contract_digest"] = "0" * 64
    payload["index_digest"] = common.serialization.canonical_json_sha256(datasets.packages.trajectory._index_digest_payload(payload))
    index_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(datasets.packages.trajectory.TransientDataContractError, match="contract or identity"):
        datasets.runtime.transient.TransientPhysicalDataset(index_path, sampling=_one_step_sampling(), source_root=tmp_path)


def test_transient_index_excludes_irregular_stop_and_derives_increments(tmp_path: Path) -> None:
    """Protect exact regular transitions, typed physical tensors, and portable identity."""
    source = tmp_path / "case.h5"
    increments = _write_transient_case(source)
    generation.publication.storage.validate_case_hdf5(source, expected_profile="transient_drying")
    index_path = tmp_path / "index.json"
    payload = _build_index(source, index_path)

    assert payload["schema_version"] == 2
    assert payload["sample_count"] == 2
    assert [(sample["time_index_n"], sample["time_index_n_plus_1"]) for sample in payload["samples"]] == [(0, 1), (1, 2)]
    assert set(payload["samples"][0]) == {
        "case_index",
        "sample_id",
        "time_index_n",
        "time_index_n_plus_1",
        "schedule_index_n",
        "schedule_index_n_plus_1",
    }
    assert payload["configured_regular_horizon"] == {
        "value": 168.0,
        "unit": "h",
        "source": "generation_scientific_config.time.stop",
    }
    assert payload["cases"][0]["sequence_length"] == 3
    assert payload["cases"][0]["stored_state_count"] == 4
    assert payload["cases"][0]["irregular_stop_time"] == 2.5
    assert payload["cases"][0]["transition_count"] == 2
    assert payload["contract"]["time"]["regular_transition_step"] == {"value": 1.0, "unit": "h"}

    dataset = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=_one_step_sampling(),
        source_root=tmp_path,
        hdf5_cache_size=1,
    )
    assert len(dataset) == 2
    first = dataset[0]
    assert first["state"].shape == (4, 251, 401)
    assert first["static"].shape == (7, 251, 401)
    assert first["boundary"].shape == (5,)
    assert first["scalars"].shape == (8,)
    torch.testing.assert_close(
        first["scalars"],
        torch.tensor([2.0e-5, 1.5, 0.4, 0.1, 0.002, 0.3, 0.2, 1300.0]),
    )
    assert first["target"].shape == (4, 251, 401)
    first_time = cast("dict[str, torch.Tensor]", first["time"])
    assert {name: value.shape for name, value in first_time.items()} == {
        "t_n": (),
        "t_n_plus_1": (),
        "dt": (),
    }
    torch.testing.assert_close(
        first["target"],
        torch.from_numpy(np.broadcast_to(increments[:, None, None], (4, 251, 401)).copy()),
    )
    assert first_time["t_n"].item() == 0.0
    assert first_time["t_n_plus_1"].item() == 1.0
    assert first_time["dt"].item() == 1.0
    assert first["metadata"]["sample_mode"] == "one_step_transition"
    assert first["metadata"]["has_exact_stop_state"] is True
    assert first["metadata"]["t_stop_exact"] == 2.5
    assert first["metadata"]["stored_state_count"] == 4
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


def test_temporal_policy_and_sampling_preserve_source_hdf5_identity(tmp_path: Path) -> None:
    """Keep training-only time choices out of source and physical HDF5 identity."""
    scientific = _synthetic_scientific_contract()
    scientific["grid"].update(
        {
            "nx": 5,
            "ny": 4,
            "Lx": 0.4,
            "Ly": 0.3,
            "Lz": 0.2,
            "dx": 0.1,
            "dy": 0.1,
        }
    )
    scientific["time"].update(
        {
            "stop": 4.0,
            "interval": 1.0,
            "regular_times": [0.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    source = tmp_path / "policy-invariant.h5"
    _write_transient_case(
        source,
        exact_stop_time=3.5,
        scientific_contract=scientific,
        regular_state_count=4,
    )
    source_sha256 = common.serialization.file_sha256(source)
    identity_names = (
        "case_input_id",
        "simulation_case_id",
        "scientific_config_digest",
        "export_contract_sha256",
    )
    with h5py.File(source, "r") as handle:
        source_identities = tuple(str(handle.attrs[name]) for name in identity_names)

    index_path = tmp_path / "policy-invariant-index.json"
    payload = _build_index(source, index_path)
    one_step = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=_one_step_sampling(),
        source_root=tmp_path,
    )
    rollout = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=datasets.contracts.transient.TransientSamplingSpec(
            mode="rollout_window",
            rollout_length=2,
            window_stride=1,
            window_offset=0,
        ),
        source_root=tmp_path,
    )

    assert one_step.dataset_id == rollout.dataset_id == payload["dataset_id"]
    assert one_step.payload["index_digest"] == rollout.payload["index_digest"] == payload["index_digest"]
    assert one_step.payload["cases"] == rollout.payload["cases"] == payload["cases"]
    assert payload["cases"][0]["source_hdf5_sha256"] == source_sha256
    assert one_step[1]["metadata"]["simulation_case_id"] == rollout[0]["metadata"]["simulation_case_id"]

    current_time = one_step[1]["time"]["t_n"]
    normalized = learning.temporal.apply_temporal_conditioning(
        current_time,
        learning.temporal.TemporalConditioningSpec(kind="normalized_current_time"),
        configured_regular_horizon=one_step.configured_regular_horizon,
    )
    disabled = learning.temporal.apply_temporal_conditioning(
        current_time,
        learning.temporal.TemporalConditioningSpec(kind="none"),
        configured_regular_horizon=one_step.configured_regular_horizon,
    )
    assert normalized is not None
    torch.testing.assert_close(normalized, torch.tensor(0.25))
    exact_stop_time = float(one_step[1]["metadata"]["t_stop_exact"])
    assert not torch.isclose(normalized, current_time / exact_stop_time)
    assert disabled is None

    with h5py.File(source, "r") as handle:
        assert tuple(str(handle.attrs[name]) for name in identity_names) == source_identities
    assert common.serialization.file_sha256(source) == source_sha256


def test_transient_rollout_windows_share_runtime_and_exclude_exact_stop(tmp_path: Path) -> None:
    """Protect deterministic aligned windows, time tensors, and regular-only endpoints."""
    scientific = _synthetic_scientific_contract()
    scientific["grid"].update(
        {
            "nx": 5,
            "ny": 4,
            "Lx": 0.4,
            "Ly": 0.3,
            "Lz": 0.2,
            "dx": 0.1,
            "dy": 0.1,
        }
    )
    scientific["time"].update(
        {
            "stop": 6.0,
            "interval": 1.0,
            "regular_times": [float(index) for index in range(7)],
        }
    )
    source = tmp_path / "rollout.h5"
    increments = _write_transient_case(
        source,
        exact_stop_time=5.5,
        scientific_contract=scientific,
        regular_state_count=6,
    )
    second_source = tmp_path / "rollout-second.h5"
    _write_transient_case(
        second_source,
        exact_stop_time=5.5,
        case_input_id="3" * 64,
        simulation_case_id="4" * 64,
        scientific_contract=scientific,
        regular_state_count=6,
    )
    index_path = tmp_path / "rollout-index.json"
    payload = datasets.packages.trajectory.build_transient_index(
        [
            _source(source),
            _source(
                second_source,
                package_case_id="synthetic_transient__case_0002",
                case_input_id="3" * 64,
                simulation_case_id="4" * 64,
            ),
        ],
        index_path,
        dataset_name="transient_drying__lentil__id",
        dataset_id="transient_drying__lentil__id__synthetic",
        evaluation_regime="id",
        source_root=tmp_path,
    )
    assert payload["transition_count"] == 10
    sampling = datasets.contracts.transient.TransientSamplingSpec(
        mode="rollout_window",
        rollout_length=2,
        window_stride=2,
        window_offset=1,
    )
    dataset = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=sampling,
        source_root=tmp_path,
        hdf5_cache_size=1,
    )

    assert dataset.configured_regular_horizon == 6.0
    assert len(dataset) == 4
    first = dataset[0]
    second = dataset[1]
    third = dataset[2]
    assert first["state"].shape == (4, 4, 5)
    assert first["static"].shape == (7, 4, 5)
    assert first["boundary"].shape == (2, 5)
    assert first["scalars"].shape == (8,)
    assert first["target"].shape == (2, 4, 4, 5)
    first_time = cast("dict[str, torch.Tensor]", first["time"])
    torch.testing.assert_close(first["state"][:, 0, 0], torch.tensor([296.0, 0.41, 109.5, 109.75]))
    torch.testing.assert_close(first["boundary"][:, :2], torch.tensor([[295.01, 295.02], [295.02, 295.03]]))
    torch.testing.assert_close(first["boundary"][0, 3], first["boundary"][1, 2])
    assert {name: value.shape for name, value in first_time.items()} == {
        "t_n": (2,),
        "t_n_plus_1": (2,),
        "dt": (2,),
    }
    torch.testing.assert_close(first_time["t_n"], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(first_time["t_n_plus_1"], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(first_time["dt"], torch.ones(2))
    torch.testing.assert_close(
        first["target"],
        torch.from_numpy(np.broadcast_to(increments[None, :, None, None], (2, 4, 4, 5)).copy()),
    )
    assert first["metadata"]["sample_mode"] == "rollout_window"
    assert first["metadata"]["rollout_length"] == 2
    assert first["metadata"]["sample_id"].endswith("__window_0001_0003")
    assert second["metadata"]["sample_id"].endswith("__window_0003_0005")
    assert first["metadata"]["simulation_case_id"] == second["metadata"]["simulation_case_id"] == "2" * 64
    assert third["metadata"]["sample_id"].endswith("case_0002__window_0001_0003")
    assert third["metadata"]["simulation_case_id"] == "4" * 64
    assert second["metadata"]["time_index_n_plus_1"] == 5
    assert second["metadata"]["has_exact_stop_state"] is True
    assert second["metadata"]["t_stop_exact"] == 5.5
    assert second["time"]["t_n_plus_1"][-1].item() == 5.0
    assert len(dataset._handles) == 1

    duplicate = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=sampling,
        source_root=tmp_path,
    )
    assert [dataset[index]["metadata"]["sample_id"] for index in range(len(dataset))] == [
        duplicate[index]["metadata"]["sample_id"] for index in range(len(duplicate))
    ]
    for num_workers in (0, 2):
        worker_dataset = datasets.runtime.transient.TransientPhysicalDataset(
            index_path,
            sampling=sampling,
            source_root=tmp_path,
            hdf5_cache_size=1,
        )
        loader = datasets.runtime.factory.make_data_loader(
            worker_dataset,
            datasets.runtime.factory.LoaderSettings(
                batch_size=2,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,
                prefetch_factor=1 if num_workers > 0 else None,
                hdf5_cache_size=1,
            ),
        )
        batch = next(iter(loader))
        assert batch["state"].shape == (2, 4, 4, 5)
        assert batch["target"].shape == (2, 2, 4, 4, 5)
        assert batch["time"]["t_n"].shape == (2, 2)
        del loader
        worker_dataset.close()
    with pytest.raises(ValueError, match="no complete rollout window"):
        datasets.runtime.transient.TransientPhysicalDataset(
            index_path,
            sampling=datasets.contracts.transient.TransientSamplingSpec(
                mode="rollout_window",
                rollout_length=6,
                window_stride=1,
                window_offset=0,
            ),
            source_root=tmp_path,
        )


def test_transient_loader_is_worker_safe_and_rejects_source_mutation(tmp_path: Path) -> None:
    """Protect zero/multi-worker collation, bounded handles, and source integrity."""
    source = tmp_path / "case.h5"
    _write_transient_case(source)
    index_path = tmp_path / "index.json"
    _build_index(source, index_path)

    zero_worker_dataset = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=_one_step_sampling(),
        source_root=tmp_path,
    )
    zero_loader = datasets.runtime.factory.make_data_loader(
        zero_worker_dataset,
        datasets.runtime.factory.LoaderSettings(batch_size=2),
    )
    zero_batch = next(iter(zero_loader))
    assert zero_batch["state"].shape == (2, 4, 251, 401)
    assert zero_batch["metadata"]["sample_id"] == [
        "synthetic_transient__case_0001__step_0000",
        "synthetic_transient__case_0001__step_0001",
    ]

    worker_dataset = datasets.runtime.transient.TransientPhysicalDataset(
        index_path,
        sampling=_one_step_sampling(),
        source_root=tmp_path,
        hdf5_cache_size=1,
    )
    _ = worker_dataset[0]
    worker_loader = datasets.runtime.factory.make_data_loader(
        worker_dataset,
        datasets.runtime.factory.LoaderSettings(
            batch_size=1,
            num_workers=2,
            persistent_workers=True,
            prefetch_factor=1,
            hdf5_cache_size=1,
        ),
    )
    worker_batch = next(iter(worker_loader))
    assert worker_batch["target"].shape == (1, 4, 251, 401)
    del worker_loader

    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="changed after dataset admission"):
        _ = worker_dataset[0]


def test_transient_hdf5_rejects_an_extra_legacy_scalar_entry(tmp_path: Path) -> None:
    """Reject an obsolete extra-entry source at canonical HDF5 admission."""
    source = tmp_path / "legacy-scalar-shape.h5"
    _write_transient_case(source)
    with h5py.File(source, "r+") as handle:
        scalar_dataset = handle["provenance/scalar_handoff_json"]
        assert isinstance(scalar_dataset, h5py.Dataset)
        raw_value = scalar_dataset[()]
        if isinstance(raw_value, bytes):
            raw_text = raw_value.decode("utf-8")
        else:
            assert isinstance(raw_value, str)
            raw_text = raw_value
        handoff = json.loads(raw_text)
        handoff["entries"].append(
            {
                "name": "retired_scalar",
                "value": 1.0,
                "unit": "1",
                "owner": "case_dependent",
            }
        )
        del handle["provenance/scalar_handoff_json"]
        provenance = handle["provenance"]
        assert isinstance(provenance, h5py.Group)
        provenance.create_dataset(
            "scalar_handoff_json",
            data=_json(handoff),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

    with pytest.raises(ValueError, match="missing, duplicate, unknown"):
        generation.publication.storage.validate_case_hdf5(
            source,
            expected_profile="transient_drying",
        )


def test_transient_time_classification_and_bad_commit(tmp_path: Path) -> None:
    """Protect no-stop, irregular-stop, regular-stop, and source-commit rejection."""
    time_contract = _synthetic_scientific_contract()["time"]
    tolerance = generation.publication.storage.time_classification_tolerance(time_contract)
    regular, positions, irregular = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        time_contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(positions, [0, 1, 2])
    assert irregular is None
    regular, _positions, irregular = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0, 2.5], dtype=np.float64),
        time_contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    assert irregular == 3
    regular, _positions, irregular = generation.publication.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0 + tolerance / 2.0], dtype=np.float64),
        time_contract,
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    assert irregular is None
    with pytest.raises(ValueError, match="optional irregular state must be final"):
        generation.publication.storage._classify_transient_times(
            np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
            time_contract,
        )

    no_exact = tmp_path / "no-exact-stop.h5"
    _write_transient_case(no_exact, exact_stop_time=None)
    generation.publication.storage.validate_case_hdf5(no_exact, expected_profile="transient_drying")
    no_exact_index = _build_index(no_exact, tmp_path / "no-exact-stop.json")
    assert no_exact_index["cases"][0]["stored_state_count"] == 3
    assert no_exact_index["cases"][0]["irregular_stop_time"] is None

    unsupported = tmp_path / "unsupported-schema.h5"
    _write_transient_case(unsupported)
    with h5py.File(unsupported, "r+") as handle:
        handle.attrs["schema_version"] = generation.publication.storage.HDF5_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported canonical case HDF5 schema"):
        generation.publication.storage.validate_case_hdf5(unsupported, expected_profile="transient_drying")

    bad = tmp_path / "bad-commit.h5"
    _write_transient_case(bad)
    with h5py.File(bad, "r+") as handle:
        handle.attrs["git_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="40-character"):
        generation.publication.storage.validate_case_hdf5(bad, expected_profile="transient_drying")
