# ruff: noqa: S101, PLR2004, SLF001
"""Physical-unit transient one-hour index and lazy-loader contracts."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
import torch

from src import common, datasets, generation

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


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
    exact_stop_time: float | None = 2.5,
    case_input_id: str = "1" * 64,
    simulation_case_id: str = "2" * 64,
) -> np.ndarray:
    """Write one compressed schema-v1 case with an optional exact-stop state."""
    x_axis = np.linspace(0.0, 1.2, 401, dtype=np.float64)
    y_axis = np.linspace(0.0, 0.75, 251, dtype=np.float64)
    shape = (y_axis.size, x_axis.size)
    static_constants = (1.0e-10, 0.0, 2.0e-10, 0.4, 100.0, 0.2, 0.1, 0.05, 101325.0, 550.0)
    assert len(static_constants) == len(generation.profiles.TRANSIENT_STATIC_FIELD_NAMES)
    static = np.stack([np.full(shape, value, dtype=np.float32) for value in static_constants])
    regular_time = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
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
        "T_flow_ref": 300.65,
        "p_ref": 101325.0,
        "p_out": 0.0,
        "T_init": 295.0,
        "T_amb": 295.0,
        "T_in_ref": 295.84,
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
        "f_wet_dm_max": 0.05,
    }
    scalar_names = generation.profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    scalars = np.asarray([scalar_values[name] for name in scalar_names], dtype=np.float64)
    ownership = ["package_fixed" if name in {"T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"} else "case_dependent" for name in scalar_names]
    schedule = np.zeros((169, len(generation.profiles.SCHEDULE_FIELDS)), dtype=np.float64)
    schedule[:, 0] = np.arange(169, dtype=np.float64)
    schedule[:, 1] = 295.0 + 0.01 * schedule[:, 0]
    schedule[:, 2] = 0.009
    schedule[:, 3] = generation.schedule.humidity_ratio_to_relative_humidity(
        schedule[:, 2],
        schedule[:, 1],
        pressure=scalar_values["p_ref"],
    )
    global_values = np.zeros((complete_time.size, len(generation.profiles.GLOBAL_FIELD_NAMES)), dtype=np.float64)
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
            0.8 * 1.2 * 0.75 * water,
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
            global_values[-1, generation.profiles.GLOBAL_FIELD_NAMES.index("f_wet_dm")],
            final_x_wb,
            final_x_wb,
            float(final_fields[0, 0, 0]),
            float(final_fields[0, 0, 0]),
            float(final_fields[1, 0, 0]),
            float(final_fields[1, 0, 0]),
        ],
        dtype=np.float64,
    )
    profile = generation.profiles.get_profile("transient_drying")
    scientific = {
        "schema_version": generation.config.CONFIG_SCHEMA_VERSION,
        "simulation_profile": "transient_drying",
    }
    scientific_digest = common.serialization.canonical_json_sha256(scientific)
    scalar_entries = [
        {"name": name, "value": scalar_values[name], "unit": unit, "owner": owner}
        for name, unit, owner in zip(
            scalar_names,
            generation.profiles.TRANSIENT_SCALAR_INPUT_UNITS,
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
        "sampled_values": scalar_values,
        "sampled_units": dict(zip(scalar_names, generation.profiles.TRANSIENT_SCALAR_INPUT_UNITS, strict=True)),
        "coupled_selections": {},
        "ood": {"natural_support_state": "natural"},
        "spatial_diagnostics": {},
        "schedule_diagnostics": {},
    }

    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            {
                "schema_kind": generation.storage.HDF5_SCHEMA_KIND,
                "schema_version": generation.storage.HDF5_SCHEMA_VERSION,
                "converter_version": generation.storage.HDF5_CONVERTER_VERSION,
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
            chunks=(1, 64, 64),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        static_dataset.attrs["field_names"] = _json(list(generation.profiles.TRANSIENT_STATIC_FIELD_NAMES))
        static_dataset.attrs["units"] = _json(list(generation.profiles.TRANSIENT_STATIC_FIELD_UNITS))
        scalar_dataset = handle.create_group("scalar").create_dataset("values", data=scalars)
        scalar_dataset.attrs["field_names"] = _json(list(scalar_names))
        scalar_dataset.attrs["units"] = _json(list(generation.profiles.TRANSIENT_SCALAR_INPUT_UNITS))
        scalar_dataset.attrs["ownership"] = _json(ownership)
        time_dataset = handle.create_dataset("time", data=regular_time)
        time_dataset.attrs["unit"] = "h"
        time_dataset.attrs["classification_atol"] = generation.storage.TIME_CLASSIFICATION_ATOL
        time_dataset.attrs["classification_basis"] = "16*float64_epsilon*168h; numerical classification only"
        transient_dataset = handle.create_group("transient").create_dataset(
            "fields",
            data=transient,
            chunks=(1, 1, 64, 64),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        transient_dataset.attrs["field_names"] = _json(list(generation.profiles.TRANSIENT_FIELD_NAMES))
        transient_dataset.attrs["units"] = _json(list(generation.profiles.TRANSIENT_FIELD_UNITS))
        if exact_fields is not None:
            exact_group = handle.create_group("exact_stop")
            exact_time_dataset = exact_group.create_dataset("time", data=np.asarray([exact_stop_time], dtype=np.float64))
            exact_time_dataset.attrs["unit"] = "h"
            exact_dataset = exact_group.create_dataset(
                "fields",
                data=exact_fields,
                chunks=(1, 64, 64),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            exact_dataset.attrs["field_names"] = _json(list(generation.profiles.TRANSIENT_FIELD_NAMES))
            exact_dataset.attrs["units"] = _json(list(generation.profiles.TRANSIENT_FIELD_UNITS))
            exact_group.attrs["usage"] = "diagnostic_only_no_training_transition"
        schedule_dataset = handle.create_group("schedule").create_dataset(
            "values",
            data=schedule,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        schedule_dataset.attrs["field_names"] = _json(list(generation.profiles.SCHEDULE_FIELDS))
        schedule_dataset.attrs["units"] = _json(list(generation.profiles.SCHEDULE_UNITS))
        schedule_dataset.attrs["conversion_pressure"] = _json(conversion_pressure)
        schedule_dataset.attrs["humidity_conversion_owner"] = "generation_schedule"
        global_dataset = handle.create_group("global").create_dataset(
            "values",
            data=global_values,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        global_dataset.attrs["field_names"] = _json(list(generation.profiles.GLOBAL_FIELD_NAMES))
        global_dataset.attrs["units"] = _json(list(generation.profiles.GLOBAL_FIELD_UNITS))
        final_dataset = handle.create_group("final_status").create_dataset("values", data=final_status)
        final_dataset.attrs["field_names"] = _json(list(generation.profiles.FINAL_STATUS_FIELDS))
        final_dataset.attrs["units"] = _json(list(generation.profiles.FINAL_STATUS_UNITS))
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
    assert first["state"].shape == (4, 251, 401)
    assert first["static"].shape == (7, 251, 401)
    assert first["boundary"].shape == (5,)
    assert first["scalars"].shape == (8,)
    assert first["target"].shape == (4, 251, 401)
    assert first["dt"].shape == ()
    torch.testing.assert_close(
        first["target"],
        torch.from_numpy(np.broadcast_to(increments[:, None, None], (4, 251, 401)).copy()),
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
    assert zero_batch["state"].shape == (2, 4, 251, 401)
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
    assert worker_batch["target"].shape == (1, 4, 251, 401)
    del worker_loader

    with source.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="changed after dataset admission"):
        _ = worker_dataset[0]


def test_transient_time_classification_and_bad_commit(tmp_path: Path) -> None:
    """Protect no-stop, irregular-stop, regular-stop, and source-commit rejection."""
    regular, positions, irregular = generation.storage._classify_transient_times(np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(positions, [0, 1, 2])
    assert irregular is None
    regular, _positions, irregular = generation.storage._classify_transient_times(np.asarray([0.0, 1.0, 2.0, 2.5], dtype=np.float64))
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    assert irregular == 3
    regular, _positions, irregular = generation.storage._classify_transient_times(
        np.asarray([0.0, 1.0, 2.0 + generation.storage.TIME_CLASSIFICATION_ATOL / 2.0], dtype=np.float64)
    )
    np.testing.assert_array_equal(regular, [0.0, 1.0, 2.0])
    assert irregular is None
    with pytest.raises(ValueError, match="optional irregular state must be the final state"):
        generation.storage._classify_transient_times(np.asarray([0.0, 0.5, 1.0], dtype=np.float64))

    no_exact = tmp_path / "no-exact-stop.h5"
    _write_transient_case(no_exact, exact_stop_time=None)
    generation.storage.validate_case_hdf5(no_exact, expected_profile="transient_drying")
    no_exact_index = _build_index(no_exact, tmp_path / "no-exact-stop.json")
    assert no_exact_index["cases"][0]["stored_state_count"] == 3
    assert no_exact_index["cases"][0]["irregular_stop_time"] is None

    unsupported = tmp_path / "unsupported-schema.h5"
    _write_transient_case(unsupported)
    with h5py.File(unsupported, "r+") as handle:
        handle.attrs["schema_version"] = generation.storage.HDF5_SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="Unsupported canonical case HDF5 schema"):
        generation.storage.validate_case_hdf5(unsupported, expected_profile="transient_drying")

    bad = tmp_path / "bad-commit.h5"
    _write_transient_case(bad)
    with h5py.File(bad, "r+") as handle:
        handle.attrs["git_commit"] = "not-a-commit"
    with pytest.raises(ValueError, match="40-character"):
        generation.storage.validate_case_hdf5(bad, expected_profile="transient_drying")
