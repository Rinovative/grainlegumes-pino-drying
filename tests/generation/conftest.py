"""Synthetic layered configuration and fake-COMSOL fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Mapping

_SMOKE_CASE_COUNT = 2
_TEST_MATERIAL_FAMILY = "lentil"
_TEST_STEADY_SEED = 41001
_TEST_TRANSIENT_SEED = 41002
_TEST_PAIRED_SEED = 41003


def _steady_flow_conditioning(
    fixed_values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return one synthetic stationary-airflow dependency audit."""
    dependencies = [
        {
            "name": name,
            "affects_stationary_solution": True,
            "owner": "model_input",
            "unit": unit,
        }
        for name, unit in (
            ("Kxx", "m^2"),
            ("Kxy", "m^2"),
            ("Kyy", "m^2"),
            ("eps_bed", "1"),
            ("p_in_bc", "Pa"),
        )
    ]
    dependencies.extend(
        {
            "name": name,
            "affects_stationary_solution": True,
            "owner": "package_fixed",
            "unit": unit,
            "fixed_value": value,
        }
        for name, unit, value in ((name, str(fixed_values[name]["unit"]), fixed_values[name]["value"]) for name in ("T_flow_ref", "p_ref", "p_out"))
    )
    return {
        "schema_kind": "steady_flow_conditioning",
        "schema_version": 1,
        "exhaustive": True,
        "stationary_solution_contract_id": "vp2_stationary_airflow_v1",
        "dependencies": dependencies,
        "additional_case_varying_solver_scalars": [],
    }


def _profile_configuration(
    simulation_profile: str,
    *,
    repeated_airflow_times: bool,
    fixed_values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return complete test-owned mappings without inspecting template binaries."""
    del repeated_airflow_times
    profile = profiles.get_profile(simulation_profile)
    patterns = {
        "steady_flow_fields": "airflow.csv",
        "transient_fields": "transient.csv",
        "global_time_series": "globals.csv",
        "final_status": "status.csv",
    }
    temporal_kinds = {
        "steady_flow_fields": "stationary",
        "transient_fields": "regular_time_series",
        "global_time_series": "regular_time_series",
        "final_status": "final_status",
    }
    exports = []
    for role in profile.export_roles:
        source = patterns[role.role]
        exports.append(
            {
                "role": role.role,
                "temporal_kind": temporal_kinds[role.role],
                "source": source,
                "delimiter": ";",
                "columns": {name: "mt.phi" if role.role == "transient_fields" and name == "phi" else name for name in role.logical_fields},
            }
        )
    return {
        "schema_kind": "generation_profile",
        "schema_version": 1,
        "simulation_profile": simulation_profile,
        "steady_flow_conditioning": _steady_flow_conditioning(fixed_values),
        "exports": exports,
    }


@pytest.fixture(autouse=True)
def _git_commit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every synthetic case and campaign to one exact fake commit."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)


@pytest.fixture
def generation_config_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return a factory for complete layered synthetic campaign configurations."""
    repository_root = Path(__file__).resolve().parents[2]
    project_root = tmp_path / "project"
    links = {
        "simulation/steady_flow/steady_flow_template.mph": repository_root / "simulation/steady_flow/steady_flow_template.mph",
        "simulation/steady_flow/steady_flow_template.sha256": repository_root / "simulation/steady_flow/steady_flow_template.sha256",
        "simulation/transient_drying/transient_drying_template.mph": repository_root / "simulation/transient_drying/transient_drying_template.mph",
        "simulation/transient_drying/transient_drying_template.sha256": repository_root
        / "simulation/transient_drying/transient_drying_template.sha256",
        "scripts/generation_campaign_node.sh": repository_root / "scripts/generation_campaign_node.sh",
    }
    for relative, source in links.items():
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            if relative == "scripts/generation_campaign_node.sh":
                shutil.copy2(source, destination)
            else:
                destination.symlink_to(source)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))

    registry = yaml.safe_load((repository_root / "configs/generation/registry.yaml").read_text(encoding="utf-8"))
    sources = yaml.safe_load((repository_root / "configs/generation/sources.yaml").read_text(encoding="utf-8"))
    materials_root = project_root / "configs/generation/materials"
    materials_root.mkdir(parents=True, exist_ok=True)
    for material_family in (_TEST_MATERIAL_FAMILY,):
        source = repository_root / "configs/generation/materials" / f"{material_family}.yaml"
        material = yaml.safe_load(source.read_text(encoding="utf-8"))
        (materials_root / f"{material_family}.yaml").write_text(
            yaml.safe_dump(material, sort_keys=False),
            encoding="utf-8",
        )

    def build(
        *,
        simulation_profile: str = "transient_drying",
        material_families: tuple[str, ...] = (_TEST_MATERIAL_FAMILY,),
        executable: Path | None = None,
        timeout: float = 5.0,
        scheduler_kind: str = "local",
        retain_solved_model: bool = False,
        retain_raw_csv: bool = False,
        repeated_airflow_times: bool = False,
        extra_arguments: tuple[str, ...] = (),
        natural_count: int = _SMOKE_CASE_COUNT,
        parameter_ood_count: int = 0,
        max_running_cases: int | None = None,
    ) -> tuple[Path, Path]:
        tests_root = project_root / "configs/generation/campaigns/test_support"
        campaign_number = len(list(tests_root.glob("campaign_*"))) if tests_root.exists() else 0
        directory = tests_root / f"campaign_{campaign_number}"
        directory.mkdir(parents=True)
        common = yaml.safe_load((repository_root / "configs/generation/common.yaml").read_text(encoding="utf-8"))
        operations = yaml.safe_load((repository_root / "configs/generation/operations/fixed_bed.yaml").read_text(encoding="utf-8"))
        profile = _profile_configuration(
            simulation_profile,
            repeated_airflow_times=repeated_airflow_times,
            fixed_values=common["scientific_fixed_values"],
        )
        execution = yaml.safe_load((repository_root / "configs/generation/execution/cluster_cpu.yaml").read_text(encoding="utf-8"))
        execution["runtime"]["timeout_seconds"] = timeout
        execution["runtime"]["extra_arguments"] = list(extra_arguments)
        execution["retention"]["technical_runtime_smoke"] = {
            "retain_raw_csv": retain_raw_csv,
            "retain_solved_model": retain_solved_model,
        }
        execution["submission"].update(
            {
                "pending_buffer": 1,
                "poll_interval_seconds": 1,
                "max_running_cases": max_running_cases,
            }
        )
        execution["cluster"].update(
            {
                "cores_per_case": 1,
                "wall_time": None,
                "scheduler_options": [],
            }
        )
        execution["site"].update(
            {
                "cpu_host": "synthetic-cpu.example",
                "scheduler": scheduler_kind,
                "partition": "test",
                "cores_per_node": 24,
                "python_module": "Python/test-fixture",
                "comsol_module": "Comsol/test-fixture",
                "python_executable": "python3",
                "comsol_executable": "comsol" if executable is None else str(executable),
            }
        )
        layers = {
            "sources.yaml": sources,
            "registry.yaml": registry,
            "common.yaml": common,
            "operations.yaml": operations,
            "profile.yaml": profile,
            "execution.yaml": execution,
        }
        for name, value in layers.items():
            (directory / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

        if material_families != (_TEST_MATERIAL_FAMILY,):
            message = "Synthetic runtime fixtures use one explicit test-owned material."
            raise ValueError(message)
        if isinstance(natural_count, bool) or not isinstance(natural_count, int) or natural_count < 1:
            message = "Synthetic runtime fixtures require at least one natural case."
            raise ValueError(message)
        if parameter_ood_count != 0:
            message = "Synthetic runtime fixtures do not support parameter OOD."
            raise ValueError(message)
        campaign_seed = _TEST_STEADY_SEED if simulation_profile == "steady_flow" else _TEST_TRANSIENT_SEED
        campaign = {
            "schema_kind": "generation_campaign",
            "schema_version": 1,
            "campaign_purpose": "technical_runtime_smoke",
            "sources_config": "sources.yaml",
            "registry_config": "registry.yaml",
            "common_config": "common.yaml",
            "operations_config": "operations.yaml",
            "profile_config": "profile.yaml",
            "execution_config": "execution.yaml",
            "paired_equivalence_seed": _TEST_PAIRED_SEED,
            "material_roles": {
                "seen": [_TEST_MATERIAL_FAMILY],
                "near_family_ood": [],
                "far_family_ood": [],
                "extreme_family_ood": [],
            },
            "sampling": {
                "method": "lhs",
                "seed_base": campaign_seed,
                "counts": {"natural": {_TEST_MATERIAL_FAMILY: natural_count}},
            },
            "dataset_packages": [
                {"evaluation_regime": "id", "source_role": "seen"},
            ],
        }
        config_path = directory / "campaign.yaml"
        config_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        return config_path, profiles.get_profile(simulation_profile).template_path

    return build


@pytest.fixture
def fake_comsol(tmp_path: Path) -> Path:
    """Return an executable emitting complete synthetic outputs for both profiles."""
    path = tmp_path / "fake_comsol.py"
    path.write_text(
        r"""#!/usr/bin/env python3
import csv
import fcntl
import json
import math
import os
import pathlib
import sys
import time


def update_tracker(delta):
    tracker = os.environ.get("FAKE_COMSOL_TRACKER")
    if not tracker:
        return
    path = pathlib.Path(tracker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        raw = stream.read().strip()
        state = {"active": 0, "maximum": 0, "starts": 0} if not raw else json.loads(raw)
        state["active"] += delta
        if delta > 0:
            state["starts"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        stream.seek(0)
        stream.truncate()
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def wait_for_expected_starts():
    expected = int(os.environ.get("FAKE_COMSOL_EXPECT_STARTS", "1"))
    if expected <= 1:
        return
    tracker = os.environ.get("FAKE_COMSOL_TRACKER")
    if not tracker:
        raise RuntimeError("FAKE_COMSOL_TRACKER is required for a start barrier")
    path = pathlib.Path(tracker)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        with path.open("r", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            state = json.load(stream)
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        if state["starts"] >= expected:
            return
        time.sleep(0.01)
    raise RuntimeError(f"timed out waiting for {expected} fake COMSOL starts")


def write_spreadsheet_header(stream, names, metadata):
    writer = csv.writer(stream, delimiter=";", lineterminator="\n")
    for key, value in metadata:
        writer.writerow([f"% {key}", value])
    writer.writerow([f"% {names[0]}", *names[1:]])


def runtime_scalar_values(arguments, case):
    handoff = case.get("scalar_handoff")
    entries = None if not isinstance(handoff, dict) else handoff.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("transient fake COMSOL requires recorded scalar handoff entries")
    runtime_entries = [entry for entry in entries if entry.get("owner") == "case_dependent"]
    if len(entries) != 12 or runtime_entries != entries:
        raise RuntimeError("fake COMSOL scalar handoff is not the canonical 12-field runtime vector")
    flags = ("-pname", "-plist", "-pindex")
    if any(arguments.count(flag) != 1 for flag in flags):
        raise RuntimeError("fake COMSOL requires exactly one pname/plist/pindex vector")
    names = arguments[arguments.index("-pname") + 1].split(",")
    values = arguments[arguments.index("-plist") + 1].split(",")
    indices = arguments[arguments.index("-pindex") + 1].split(",")
    expected_names = [entry["name"] for entry in runtime_entries]
    expected_indices = [str(index) for index in range(1, 13)]
    if names != expected_names or indices != expected_indices or len(values) != 12:
        raise RuntimeError("fake COMSOL received a misordered or incomplete runtime scalar vector")
    parsed = {}
    for encoded, entry in zip(values, runtime_entries):
        unit = entry["unit"]
        if unit == "1":
            if "[" in encoded or "]" in encoded:
                raise RuntimeError("dimensionless fake COMSOL parameters cannot carry [1]")
            number_text = encoded
        else:
            suffix = f"[{unit}]"
            if not encoded.endswith(suffix):
                raise RuntimeError(f"fake COMSOL parameter {entry['name']} lacks unit {unit}")
            number_text = encoded[: -len(suffix)]
        number = float(number_text)
        if not math.isfinite(number) or format(number, ".17g") != number_text:
            raise RuntimeError(f"fake COMSOL parameter {entry['name']} is not canonical finite text")
        if number != float(entry["value"]):
            raise RuntimeError(f"fake COMSOL parameter {entry['name']} disagrees with case provenance")
        parsed[entry["name"]] = number
    return parsed


if sys.argv[1:] == ["-version"]:
    print("COMSOL Multiphysics 6.4.0.293")
    raise SystemExit(0)

mode = os.environ.get("FAKE_COMSOL_MODE", "success")
if mode == "failure":
    print("synthetic failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "license_capacity":
    print("Could not obtain license for 'Brinkman Equations (br)'.")
    print("Required product: CFD Module.")
    print("License error: -4.", file=sys.stderr)
    print("Licensed number of users already reached.", file=sys.stderr)
    print("Feature: COMSOL", file=sys.stderr)
    print("FlexNet Licensing error:-4,132", file=sys.stderr)
    raise SystemExit(0)
update_tracker(1)
try:
    wait_for_expected_starts()
    if mode == "timeout":
        time.sleep(2.0)
    else:
        time.sleep(float(os.environ.get("FAKE_COMSOL_DELAY", "0")))
        arguments = sys.argv[1:]
        if arguments.count("-job") != 1 or arguments[arguments.index("-job") + 1] != "b1":
            raise RuntimeError("fake COMSOL requires the canonical b1 job configuration")
        if arguments.count("-inputfile") != 1 or arguments[arguments.index("-inputfile") + 1] != "model.mph":
            raise RuntimeError("fake COMSOL requires the canonical work model")
        retained = arguments.count("-outputfile") == 1
        no_save = arguments.count("-nosave") == 1
        if retained == no_save:
            raise RuntimeError("fake COMSOL requires exactly one canonical save mode")
        case = json.loads(pathlib.Path("case.json").read_text(encoding="utf-8"))
        transient_profile = case["simulation_profile"] == "transient_drying"
        scalars = runtime_scalar_values(arguments, case) if transient_profile else {}
        solved_model_mode = os.environ.get("FAKE_COMSOL_SOLVED_MODEL_MODE", "canonical")
        if retained:
            requested_output = arguments[arguments.index("-outputfile") + 1]
            solved_model_outputs = {
                "canonical": (requested_output,),
                "suffixed": ("solved_1.mph",),
                "multiple": ("solved_1.mph", "solved_2.mph"),
                "missing": (),
                "empty": (requested_output,),
                "symlink": (requested_output,),
            }
            if solved_model_mode not in solved_model_outputs:
                raise RuntimeError(f"unsupported fake solved-model mode: {solved_model_mode}")
            selected_solved_model_outputs = solved_model_outputs[solved_model_mode]
        else:
            selected_solved_model_outputs = ()
        for solved_model_output in selected_solved_model_outputs:
            solved_model_path = pathlib.Path(solved_model_output)
            if solved_model_mode == "symlink":
                solved_model_path.symlink_to("model.mph")
            else:
                payload = b"" if solved_model_mode == "empty" else b"synthetic solved model\n"
                solved_model_path.write_bytes(payload)
        with pathlib.Path("fields.csv").open(encoding="utf-8", newline="") as stream:
            inputs = list(csv.DictReader(stream, delimiter=";"))
        x_values = [float(source["x"]) for source in inputs]
        y_values = [float(source["y"]) for source in inputs]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        rho_values = None
        cell_weights = None
        weighted_dry = None
        if transient_profile:
            rho_values = [
                scalars["rho_bu_dry_ref"] * (1.0 - float(source["eps_bed"])) / (1.0 - scalars["eps_bed_cal_ref"])
                for source in inputs
            ]
            cell_weights = [
                (0.5 if x in {x_min, x_max} else 1.0) * (0.5 if y in {y_min, y_max} else 1.0)
                for x, y in zip(x_values, y_values)
            ]
            weighted_dry = sum(rho * weight for rho, weight in zip(rho_values, cell_weights))
        exports = pathlib.Path("exports")
        exports.mkdir(exist_ok=True)
        mapping_export_mode = os.environ.get("FAKE_COMSOL_MAPPING_EXPORT_MODE", "canonical")
        if mapping_export_mode not in {"canonical", "mismatch", "missing"}:
            raise RuntimeError(f"unsupported fake mapping export mode: {mapping_export_mode}")
        repeated = os.environ.get("FAKE_COMSOL_REPEAT_AIRFLOW") == "1"
        varying = os.environ.get("FAKE_COMSOL_VARY_AIRFLOW") == "1"
        static_names = ["x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc"]
        if transient_profile:
            static_names.append("X_0_db_field")
        static_names.extend(("u", "v", "p"))
        if transient_profile:
            static_names.append("rho_bu_dry")
        airflow_path = exports / "airflow.csv"
        with airflow_path.open("w", encoding="utf-8", newline="") as stream:
            names = (["stationary_time"] if repeated else []) + static_names
            writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
            stationary_times = [0.0, 1.0] if repeated else [None]
            header_names = ["wrong_x" if name == "x" and mapping_export_mode == "mismatch" else name for name in names]
            write_spreadsheet_header(
                stream,
                header_names,
                [
                    ("Model", "model.mph"),
                    ("Version", "COMSOL 6.4.0.293"),
                    ("Dimension", 2),
                    ("Nodes", len(inputs) * len(stationary_times)),
                    ("Expressions", len(names)),
                    ("Length unit", "m"),
                ],
            )
            for stationary_time in stationary_times:
                for row_index, source in enumerate(inputs):
                    eps = float(source["eps_bed"])
                    values = {
                        **source,
                        "x": float(source["x"]) + 1e-13,
                        "y": float(source["y"]) + 1e-13,
                        "u": 0.1 + row_index * 1e-7,
                        "v": 0.2 + row_index * 1e-7,
                        "p": 10.0 + row_index * 1e-5,
                    }
                    if transient_profile:
                        values["rho_bu_dry"] = scalars["rho_bu_dry_ref"] * (1.0 - eps) / (1.0 - scalars["eps_bed_cal_ref"])
                    if varying and stationary_time == 1.0 and row_index == 0:
                        values["p"] = 99.0
                    if stationary_time is not None:
                        values["stationary_time"] = stationary_time
                    writer.writerow(values)
        if mapping_export_mode == "missing":
            airflow_path.unlink()
        if transient_profile:
            state_times = (0.0, 1.0, 1.5)
            water_by_time = {state_time: [] for state_time in state_times}
            with (exports / "transient.csv").open("w", encoding="utf-8", newline="") as stream:
                logical_units = (
                    ("t", "h"),
                    ("x", "m"),
                    ("y", "m"),
                    ("T", "K"),
                    ("mt.phi", "1"),
                    ("w_surf", "kg/m^3"),
                    ("w_int", "kg/m^3"),
                )
                names = [
                    f"{name} ({unit}) @ t={state_time:g}"
                    for state_time in state_times
                    for name, unit in logical_units
                ]
                writer = csv.writer(stream, delimiter=";", lineterminator="\n")
                write_spreadsheet_header(
                    stream,
                    names,
                    [
                        ("Model", "model.mph"),
                        ("Dimension", 2),
                        ("Nodes", len(inputs)),
                        ("Expressions", len(names)),
                    ],
                )
                for source, rho in zip(inputs, rho_values):
                    row = []
                    for state_time in state_times:
                        initial_water = rho * float(source["X_0_db_field"])
                        w_surf = initial_water - 0.2 * state_time
                        w_int = initial_water - 0.1 * state_time
                        water = scalars["f_surf"] * w_surf + (1.0 - scalars["f_surf"]) * w_int
                        water_by_time[state_time].append(water)
                        row.extend(
                            (
                                state_time,
                                float(source["x"]) + 1e-13,
                                float(source["y"]) + 1e-13,
                                296.0 - 0.1 * state_time,
                                0.5 - 0.01 * state_time,
                                w_surf,
                                w_int,
                            )
                        )
                    writer.writerow(row)
            with (exports / "globals.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "T_amb (K)",
                    "Time",
                    "t (h)",
                    "X_wb_bulk (1)",
                    "f_wet_dm (1)",
                    "m_w_gr (kg)",
                    "m_v_gas (kg)",
                    "m_dot_evap (kg/s)",
                    "m_dot_v_in (kg/s)",
                    "m_dot_v_out (kg/s)",
                    "mt_mass_balance (kg/s)",
                    "T_out_mean (K)",
                    "phi_out_mean (1)",
                ]
                writer = csv.writer(stream, delimiter=";", lineterminator="\n")
                write_spreadsheet_header(
                    stream,
                    names,
                    [("Model", "model.mph"), ("Expressions", len(names))],
                )
                for state_time in state_times:
                    water_values = water_by_time[state_time]
                    weighted_water = sum(water * weight for water, weight in zip(water_values, cell_weights))
                    bulk = weighted_water / (weighted_dry + weighted_water)
                    writer.writerow(
                        [
                            scalars["T_amb"],
                            state_time,
                            state_time,
                            bulk,
                            1.0 if state_time == 0.0 else 0.04,
                            0.8 * 9e-6 * weighted_water,
                            1.0 + 0.1 * state_time,
                            0.1 - 0.01 * state_time,
                            0.02,
                            0.01 + 0.005 * state_time,
                            0.0,
                            296.0 - 0.1 * state_time,
                            0.5 - 0.01 * state_time,
                        ]
                    )
            with (exports / "status.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "T_amb (K)",
                    "Time",
                    "t_final (h)",
                    "f_wet_dm_final (1)",
                    "X_wb_bulk_final (1)",
                    "X_wb_max_final (1)",
                    "T_min_final (K)",
                    "T_max_final (K)",
                    "phi_min_final (1)",
                    "phi_max_final (1)",
                ]
                writer = csv.writer(stream, delimiter=";", lineterminator="\n")
                write_spreadsheet_header(stream, names, [("Model", "model.mph"), ("Expressions", len(names))])
                final_time = state_times[-1]
                final_water = water_by_time[final_time]
                weighted_water = sum(water * weight for water, weight in zip(final_water, cell_weights))
                bulk = weighted_water / (weighted_dry + weighted_water)
                maximum = max(water / (rho + water) for water, rho in zip(final_water, rho_values))
                writer.writerow(
                    [
                        scalars["T_amb"],
                        0.0,
                        final_time,
                        0.04,
                        bulk,
                        maximum,
                        296.0 - 0.1 * final_time,
                        296.0 - 0.1 * final_time,
                        0.5 - 0.01 * final_time,
                        0.5 - 0.01 * final_time,
                    ]
                )
        print("synthetic success")
finally:
    update_tracker(-1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path
