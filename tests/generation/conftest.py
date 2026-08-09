"""Synthetic layered configuration and fake-COMSOL fixtures."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import generation


def _parameter_values(definitions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return compact test-only values for every final registry definition."""
    intervals: dict[str, tuple[float, float, float, float]] = {
        "kappa_mean": (4e-9, 6e-9, 7e-9, 8e-9),
        "kappa_cv": (0.2, 0.5, 0.6, 0.8),
        "bed.structure.coarse_len_rel": (0.02, 0.03, 0.035, 0.04),
        "bed.structure.fine_len_rel": (0.01, 0.02, 0.025, 0.03),
        "bed.structure.coarse_weight": (0.3, 0.7, 0.75, 0.8),
        "bed.structure.fine_ani_x": (0.8, 1.2, 1.3, 1.5),
        "bed.structure.fine_ani_y": (0.8, 1.2, 1.3, 1.5),
        "bed.structure.cross_scale_corr": (0.2, 0.6, 0.7, 0.8),
        "bed.perturbations.amplitude": (0.0, 0.01, 0.02, 0.03),
        "bed.perturbations.granularity": (0.3, 0.6, 0.7, 0.8),
        "bed.perturbations.sign_bias": (0.3, 0.7, 0.75, 0.8),
        "permeability.anisotropy.max_ratio": (1.2, 1.8, 1.9, 2.1),
        "permeability.anisotropy.exponent": (1.0, 2.0, 2.2, 2.5),
        "permeability.anisotropy.strength": (0.3, 0.8, 0.9, 1.0),
        "permeability.orientation.jitter": (0.0, 0.01, 0.02, 0.03),
        "permeability.orientation.smooth_len_rel": (0.01, 0.02, 0.025, 0.03),
        "porosity.anchor_rel": (1.5, 2.0, 2.2, 2.5),
        "porosity.smooth_len_rel": (0.01, 0.02, 0.025, 0.03),
        "porosity.texture_amp": (0.001, 0.002, 0.003, 0.004),
        "pressure_bc.mean": (300.0, 400.0, 420.0, 450.0),
        "pressure_bc.sin_amp": (0.0, 0.03, 0.04, 0.05),
        "pressure_bc.sin_freq": (0.5, 1.0, 1.1, 1.3),
        "pressure_bc.sin_phase": (0.0, 3.0, 3.5, 4.0),
        "pressure_bc.gauss_amp": (0.0, 0.03, 0.04, 0.05),
        "pressure_bc.gauss_width": (0.03, 0.05, 0.06, 0.08),
        "pressure_bc.gauss_jitter": (0.0, 0.2, 0.25, 0.3),
        "pressure_bc.linear_amp": (0.0, 0.02, 0.03, 0.04),
        "initial_moisture.mean_db": (0.19, 0.21, 0.22, 0.23),
        "initial_moisture.amplitude_db": (0.01, 0.02, 0.025, 0.03),
        "initial_moisture.structure.coarse_len_rel": (0.02, 0.03, 0.035, 0.04),
        "initial_moisture.structure.fine_len_rel": (0.01, 0.02, 0.025, 0.03),
        "initial_moisture.structure.coarse_weight": (0.3, 0.7, 0.75, 0.8),
        "initial_moisture.structure.fine_ani_x": (0.8, 1.2, 1.3, 1.5),
        "initial_moisture.structure.fine_ani_y": (0.8, 1.2, 1.3, 1.5),
        "initial_moisture.structure.cross_scale_corr": (0.2, 0.6, 0.7, 0.8),
        "T_in_base": (295.0, 297.0, 298.0, 299.0),
        "T_in_amp": (0.0, 0.5, 0.6, 0.8),
        "omega_in_base": (0.008, 0.009, 0.0095, 0.010),
        "omega_in_amp": (0.0, 0.0002, 0.0003, 0.0004),
        "schedule.corr": (-0.5, 0.5, 0.6, 0.7),
        "schedule.timescale_rel": (0.2, 0.4, 0.45, 0.6),
        "schedule.event_duration_rel": (0.02, 0.05, 0.06, 0.08),
        "schedule.event_width_rel": (0.005, 0.01, 0.012, 0.02),
        "rho_bu_dry_ref": (500.0, 600.0, 620.0, 680.0),
        "k_gr": (0.1, 0.2, 0.22, 0.3),
        "cp_gr_dry": (1000.0, 1300.0, 1400.0, 1600.0),
        "r_surf_0": (1e-5, 2e-5, 2.2e-5, 3e-5),
        "r_int_surf": (0.5, 1.5, 1.7, 2.0),
        "f_surf": (0.3, 0.7, 0.75, 0.8),
        "T_amb": (293.0, 295.0, 296.0, 297.0),
    }
    integers = {"pressure_bc.gauss_count": (1, 2, 3, 3), "schedule.event_count": (0, 2, 3, 3)}
    fixed = {
        "eps_min_global": 0.3,
        "eps_max_global": 0.8,
        "eps_bed_cal_ref": 0.5,
        "X_target_wb": 0.12,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        kind = definition["kind"]
        if kind == "interval":
            lower, upper, ood_lower, ood_upper = intervals[name]
            result[name] = {"lower": lower, "upper": upper}
            if "ood_group" in definition:
                result[name]["ood"] = {"lower": ood_lower, "upper": ood_upper}
        elif kind == "integer":
            lower, upper, ood_lower, ood_upper = integers[name]
            result[name] = {"lower": lower, "upper": upper}
            if "ood_group" in definition:
                result[name]["ood"] = {"lower": ood_lower, "upper": ood_upper}
        elif kind == "fixed":
            result[name] = {"value": fixed[name]}
        elif kind == "simplex":
            result[name] = {"ood_values": [{"smooth": 1.0, "event": 0.0, "trend": 0.0}]}
        elif kind == "parameter_set":
            result[name] = {
                "sets": [{"id": "synthetic_oswin_id", "values": {"A_osw": 0.1, "B_osw": 0.2, "C_osw": 0.3}}],
                "ood_sets": [{"id": "synthetic_oswin_ood", "values": {"A_osw": 0.4, "B_osw": 0.5, "C_osw": 0.6}}],
            }
        else:
            result[name] = {}
    return result


def _steady_flow_conditioning() -> dict[str, Any]:
    """Return one explicit synthetic stationary-airflow dependency audit."""
    model_inputs = {
        "Kxx": "m^2",
        "Kxy": "m^2",
        "Kyy": "m^2",
        "eps_bed": "1",
        "p_bc": "Pa",
    }
    dependencies = [
        {
            "name": name,
            "affects_stationary_solution": True,
            "owner": "model_input",
            "unit": unit,
            "fixed_value": None,
        }
        for name, unit in model_inputs.items()
    ]
    dependencies.append(
        {
            "name": "air_dynamic_viscosity",
            "affects_stationary_solution": True,
            "owner": "package_fixed",
            "unit": "Pa*s",
            "fixed_value": 1.8139e-5,
        }
    )
    dependencies.extend(
        {
            "name": name,
            "affects_stationary_solution": False,
            "owner": "not_used",
            "unit": unit,
            "fixed_value": None,
        }
        for name, unit in (
            ("air_density", "kg/m^3"),
            ("T_flow_ref", "K"),
            ("profile_reference_temperature", "K"),
        )
    )
    return {
        "schema_kind": "steady_flow_conditioning",
        "schema_version": 1,
        "exhaustive": True,
        "stationary_solution_contract_id": "synthetic_brinkman_airflow_v1",
        "dependencies": dependencies,
        "additional_case_varying_solver_scalars": [],
    }


def _profile_configuration(simulation_profile: str, *, repeated_airflow_times: bool) -> dict[str, Any]:
    """Return complete test-owned mappings without inspecting template binaries."""
    profile = generation.profiles.get_profile(simulation_profile)
    patterns = {
        "steady_flow_fields": "airflow.csv",
        "transient_fields": "transient.csv",
        "global_time_series": "globals.csv",
        "final_status": "status.csv",
    }
    exports = [
        {
            "role": role.role,
            "pattern": patterns[role.role],
            "delimiter": ";",
            "columns": {name: name for name in role.logical_fields},
            "time_column": "stationary_time" if role.role == "steady_flow_fields" and repeated_airflow_times else None,
        }
        for role in profile.export_roles
    ]
    return {
        "schema_kind": "generation_profile",
        "schema_version": 1,
        "simulation_profile": simulation_profile,
        "template_ready": True,
        "steady_flow_conditioning": _steady_flow_conditioning(),
        "exports": exports,
    }


def _resolved_evidence() -> dict[str, Any]:
    """Return explicit test-only evidence that cannot be mistaken for literature."""
    return {
        "source": "synthetic pytest fixture; not a scientific source",
        "evidence_type": "assumed",
        "confidence": "low",
        "temperature_range": None,
        "humidity_range": None,
        "cultivar_or_market_class": "synthetic",
        "product_form": "synthetic",
        "status": "resolved",
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
        "simulation/steady_flow/template_brinkman.mph": repository_root / "simulation/steady_flow/template_brinkman.mph",
        "simulation/transient_drying/template_brinkman_temp_moist.mph": repository_root
        / "simulation/transient_drying/template_brinkman_temp_moist.mph",
        "simulation/transient_drying/template.sha256": repository_root / "simulation/transient_drying/template.sha256",
        "scripts/generation_campaign_node.sh": repository_root / "scripts/generation_campaign_node.sh",
    }
    for relative, source in links.items():
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))

    registry = yaml.safe_load((repository_root / "configs/generation/registry.yaml").read_text(encoding="utf-8"))
    parameter_values = _parameter_values(registry["parameters"])
    materials_root = project_root / "configs/generation/materials"
    materials_root.mkdir(parents=True, exist_ok=True)
    for material_family in generation.materials.MATERIAL_FAMILIES:
        source = repository_root / "configs/generation/materials" / f"{material_family}.yaml"
        material = yaml.safe_load(source.read_text(encoding="utf-8"))
        material["executable"] = True
        material["taxonomy"]["specificity_status"] = "resolved"
        material["product_form"]["specificity_status"] = "resolved"
        owned_names = tuple(name for name in material["parameter_values"] if name != "initial_moisture_bounds")
        material["parameter_values"] = {
            "initial_moisture_bounds": {"lower": 0.1, "upper": 0.3},
            **{name: copy.deepcopy(parameter_values[name]) for name in owned_names},
        }
        material["evidence"] = {name: _resolved_evidence() for name in material["parameter_values"]}
        (materials_root / f"{material_family}.yaml").write_text(
            yaml.safe_dump(material, sort_keys=False),
            encoding="utf-8",
        )

    def build(
        *,
        simulation_profile: str = "transient_drying",
        material_families: tuple[str, ...] = ("lentil",),
        executable: Path | None = None,
        timeout: float = 5.0,
        scheduler_kind: str = "local",
        retain_solved_model: bool = False,
        retain_raw_csv: bool = False,
        repeated_airflow_times: bool = False,
        natural_count: int = 1,
        parameter_ood_count: int = 4,
    ) -> tuple[Path, Path]:
        tests_root = project_root / "configs/generation/campaigns/test_support"
        campaign_number = len(list(tests_root.glob("campaign_*"))) if tests_root.exists() else 0
        directory = tests_root / f"campaign_{campaign_number}"
        directory.mkdir(parents=True)
        common = yaml.safe_load((repository_root / "configs/generation/common.yaml").read_text(encoding="utf-8"))
        common["executable"] = True
        common["scientific_fixed_values"].update(
            {"p_ref": 101325.0, "omega_min": 0.001, "omega_max": 0.02, "phi_clip_min": 0.05, "phi_clip_max": 0.95}
        )
        common["parameter_values"] = {name: copy.deepcopy(parameter_values[name]) for name in common["parameter_values"]}
        operations = yaml.safe_load((repository_root / "configs/generation/operations/fixed_bed.yaml").read_text(encoding="utf-8"))
        operations["executable"] = True
        operations["parameter_values"] = {name: copy.deepcopy(parameter_values[name]) for name in operations["parameter_values"]}
        profile = _profile_configuration(simulation_profile, repeated_airflow_times=repeated_airflow_times)
        execution = yaml.safe_load((repository_root / "configs/generation/execution/cluster_cpu.yaml").read_text(encoding="utf-8"))
        execution["runtime"].update(
            {
                "executable": "comsol" if executable is None else str(executable),
                "module_initialization": [],
                "timeout_seconds": timeout,
            }
        )
        execution["retention"] = {
            "retain_raw_csv": retain_raw_csv,
            "retain_solved_model": retain_solved_model,
        }
        execution["cluster"].update(
            {
                "max_nodes": 2,
                "cases_per_node": 2,
                "cores_per_case": 1,
                "max_parallel_cases": 3,
                "cores_per_node": 32,
                "scheduler_kind": scheduler_kind,
                "partition": "test",
                "wall_time": None,
                "scheduler_options": [],
            }
        )
        execution["site"].update(
            {
                "cpu_host": "synthetic-cpu.example",
                "scheduler": scheduler_kind,
                "partition": "test",
                "cores_per_node": 32,
                "python_module": "Python/3.10",
                "comsol_module": "Comsol/v6.4",
                "python_executable": "python3",
                "comsol_executable": "comsol",
            }
        )
        layers = {
            "registry.yaml": registry,
            "common.yaml": common,
            "operations.yaml": operations,
            "profile.yaml": profile,
            "execution.yaml": execution,
        }
        for name, value in layers.items():
            (directory / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

        selected = tuple(material_family for material_family in generation.materials.MATERIAL_FAMILIES if material_family in material_families)
        if selected != material_families:
            message = "Synthetic material_families must be unique and in canonical order."
            raise ValueError(message)
        seen = tuple(material_family for material_family in selected if material_family in generation.materials.MATERIAL_FAMILIES[:3])
        near = tuple(material_family for material_family in selected if material_family == "field_pea")
        far = tuple(material_family for material_family in selected if material_family == "almond")
        dataset_views = ("steady_flow",) if simulation_profile == "steady_flow" else ("steady_flow", "transient_drying")
        dataset_packages: list[dict[str, Any]] = []
        for dataset_view in dataset_views:
            dataset_packages.extend(
                [
                    {
                        "dataset_view": dataset_view,
                        "evaluation_regime": "id",
                        "materials": list(seen),
                        "membership_seed": 9101,
                        "membership_counts_per_material": {"train": 1, "validation": 1, "id_test": 1},
                    },
                    {
                        "dataset_view": dataset_view,
                        "evaluation_regime": "parameter_ood",
                        "materials": list(seen),
                    },
                ]
            )
            for regime, role_materials in (("near_family_ood", near), ("far_family_ood", far)):
                if role_materials:
                    dataset_packages.append(
                        {
                            "dataset_view": dataset_view,
                            "evaluation_regime": regime,
                            "materials": list(role_materials),
                        }
                    )
        campaign = {
            "schema_kind": "generation_campaign",
            "schema_version": 1,
            "campaign_name": f"synthetic_{simulation_profile}_{directory.name}",
            "executable": True,
            "registry_config": "registry.yaml",
            "common_config": "common.yaml",
            "operations_config": "operations.yaml",
            "profile_config": "profile.yaml",
            "execution_config": "execution.yaml",
            "materials": list(selected),
            "roles": {"seen": list(seen), "near_family_ood": list(near), "far_family_ood": list(far)},
            "duplicate_case_input_policy": "reject_duplicates",
            "sampling": {
                "method": "lhs",
                "seed_base": 3001,
                "counts": {
                    "natural": dict.fromkeys(selected, natural_count),
                    "parameter_ood": dict.fromkeys(seen, parameter_ood_count),
                },
                "parameter_ood": {
                    "groups": list(generation.materials.OOD_GROUPS),
                    "units_per_case": 1,
                    "balance_groups": True,
                    "balance_parameters": True,
                },
            },
            "dataset_packages": dataset_packages,
        }
        config_path = directory / "campaign.yaml"
        config_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        return config_path, generation.profiles.get_profile(simulation_profile).template_path

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


def scalar_values():
    with pathlib.Path("scalars.csv").open(encoding="utf-8", newline="") as stream:
        return {row["name"]: float(row["value"]) for row in csv.DictReader(stream, delimiter=";")}


mode = os.environ.get("FAKE_COMSOL_MODE", "success")
if mode == "failure":
    print("synthetic failure", file=sys.stderr)
    raise SystemExit(7)
update_tracker(1)
try:
    if mode == "timeout":
        time.sleep(2.0)
    else:
        time.sleep(float(os.environ.get("FAKE_COMSOL_DELAY", "0")))
        arguments = sys.argv[1:]
        pathlib.Path(arguments[arguments.index("-outputfile") + 1]).write_bytes(b"synthetic solved model\n")
        case = json.loads(pathlib.Path("case.json").read_text(encoding="utf-8"))
        with pathlib.Path("fields.csv").open(encoding="utf-8", newline="") as stream:
            inputs = list(csv.DictReader(stream, delimiter=";"))
        scalars = scalar_values()
        x_values = [float(source["x"]) for source in inputs]
        y_values = [float(source["y"]) for source in inputs]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        rho_values = [
            scalars["rho_bu_dry_ref"] * (1.0 - float(source["eps_bed"])) / (1.0 - scalars["eps_bed_cal_ref"])
            for source in inputs
        ]
        cell_weights = [
            (0.5 if x in {x_min, x_max} else 1.0) * (0.5 if y in {y_min, y_max} else 1.0)
            for x, y in zip(x_values, y_values)
        ]
        weighted_dry = sum(rho * weight for rho, weight in zip(rho_values, cell_weights))
        weighted_cells = sum(cell_weights)
        exports = pathlib.Path("exports")
        exports.mkdir(exist_ok=True)
        repeated = os.environ.get("FAKE_COMSOL_REPEAT_AIRFLOW") == "1"
        varying = os.environ.get("FAKE_COMSOL_VARY_AIRFLOW") == "1"
        static_names = ["x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_bc", "X_0_db_field", "u", "v", "p", "rho_bu_dry"]
        with (exports / "airflow.csv").open("w", encoding="utf-8", newline="") as stream:
            names = (["stationary_time"] if repeated else []) + static_names
            writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
            writer.writeheader()
            for stationary_time in ([0.0, 1.0] if repeated else [None]):
                for row_index, source in enumerate(inputs):
                    eps = float(source["eps_bed"])
                    values = {
                        **source,
                        "x": float(source["x"]) + 1e-13,
                        "y": float(source["y"]) + 1e-13,
                        "u": 0.1 + row_index * 1e-7,
                        "v": 0.2 + row_index * 1e-7,
                        "p": 10.0 + row_index * 1e-5,
                        "rho_bu_dry": scalars["rho_bu_dry_ref"] * (1.0 - eps) / (1.0 - scalars["eps_bed_cal_ref"]),
                    }
                    if varying and stationary_time == 1.0 and row_index == 0:
                        values["p"] = 99.0
                    if stationary_time is not None:
                        values["stationary_time"] = stationary_time
                    writer.writerow(values)
        if case["simulation_profile"] == "transient_drying":
            with (exports / "transient.csv").open("w", encoding="utf-8", newline="") as stream:
                names = ["x", "y", "t", "T", "phi", "w_surf", "w_int"]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                for time in (0.0, 1.0):
                    for source in inputs:
                        writer.writerow(
                            {
                                "x": float(source["x"]) + 1e-13,
                                "y": float(source["y"]) + 1e-13,
                                "t": time,
                                "T": 296.0 - 0.1 * time,
                                "phi": 0.5 - 0.01 * time,
                                "w_surf": 10.0 - 0.1 * time,
                                "w_int": 20.0 - 0.1 * time,
                            }
                        )
            with (exports / "globals.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "t",
                    "X_wb_bulk",
                    "X_wb_max",
                    "X_wb_q95_mass",
                    "f_wet_dm",
                    "T_out_mean",
                    "phi_out_mean",
                    "m_w_gr",
                    "m_v_gas",
                    "m_dot_evap",
                    "m_dot_v_in",
                    "m_dot_v_out",
                ]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                for time in (0.0, 1.0):
                    water_density = 30.0 - 0.2 * time
                    bulk = water_density * weighted_cells / (weighted_dry + water_density * weighted_cells)
                    writer.writerow(
                        dict(
                            zip(
                                names,
                                [
                                    time,
                                    bulk,
                                    bulk + 0.01,
                                    bulk + 0.005,
                                    1.0 if time == 0.0 else 0.04,
                                    295.0 + 0.5 * time,
                                    0.5 - 0.05 * time,
                                    water_density * weighted_cells * 9e-6,
                                    1.0 + 0.1 * time,
                                    0.1 - 0.01 * time,
                                    0.02,
                                    0.01 + 0.005 * time,
                                ],
                            )
                        )
                    )
            with (exports / "status.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "t_final",
                    "f_wet_dm_final",
                    "X_target_wb",
                    "X_wb_bulk",
                    "X_wb_max",
                    "X_wb_q95_mass",
                    "T_min_final",
                    "T_max_final",
                    "phi_min_final",
                    "phi_max_final",
                ]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                water_density = 29.8
                bulk = water_density * weighted_cells / (weighted_dry + water_density * weighted_cells)
                writer.writerow(
                    dict(
                        zip(
                            names,
                            [1.0, 0.04, scalars["X_target_wb"], bulk, bulk + 0.01, bulk + 0.005, 294.0, 297.0, 0.3, 0.7],
                        )
                    )
                )
        print("synthetic success")
finally:
    update_tracker(-1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path
