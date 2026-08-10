"""Synthetic layered configuration and fake-COMSOL fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src import generation

_SMOKE_CASE_COUNT = 2


def _steady_flow_conditioning() -> dict[str, Any]:
    """Return one explicit synthetic stationary-airflow dependency audit."""
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
        for name, unit, value in (
            ("T_flow_ref", "K", 300.65),
            ("p_ref", "Pa", 101325.0),
            ("p_out", "Pa", 0.0),
        )
    )
    return {
        "schema_kind": "steady_flow_conditioning",
        "schema_version": 1,
        "exhaustive": True,
        "stationary_solution_contract_id": "vp2_stationary_airflow_v1",
        "dependencies": dependencies,
        "additional_case_varying_solver_scalars": [],
    }


def _profile_configuration(simulation_profile: str, *, repeated_airflow_times: bool) -> dict[str, Any]:
    """Return complete test-owned mappings without inspecting template binaries."""
    del repeated_airflow_times
    profile = generation.profiles.get_profile(simulation_profile)
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
        "exact_stop_diagnostics": "irregular_stop_diagnostic",
    }
    exports = []
    for role in profile.export_roles:
        source = (
            {"state": "mapping_probe_required"}
            if role.role == "exact_stop_diagnostics"
            else {"state": "runtime_confirmed", "pattern": patterns[role.role]}
        )
        exports.append(
            {
                "role": role.role,
                "temporal_kind": temporal_kinds[role.role],
                "source": source,
                "delimiter": ";",
                "columns": {name: {"state": "runtime_confirmed", "source_header": name} for name in role.logical_fields},
            }
        )
    return {
        "schema_kind": "generation_profile",
        "schema_version": 1,
        "simulation_profile": simulation_profile,
        "steady_flow_conditioning": _steady_flow_conditioning(),
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
            destination.symlink_to(source)
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))

    registry = yaml.safe_load((repository_root / "configs/generation/registry.yaml").read_text(encoding="utf-8"))
    sources = yaml.safe_load((repository_root / "configs/generation/sources.yaml").read_text(encoding="utf-8"))
    materials_root = project_root / "configs/generation/materials"
    materials_root.mkdir(parents=True, exist_ok=True)
    for material_family in generation.materials.MATERIAL_FAMILIES:
        source = repository_root / "configs/generation/materials" / f"{material_family}.yaml"
        material = yaml.safe_load(source.read_text(encoding="utf-8"))
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
        natural_count: int = _SMOKE_CASE_COUNT,
        parameter_ood_count: int = 0,
    ) -> tuple[Path, Path]:
        tests_root = project_root / "configs/generation/campaigns/test_support"
        campaign_number = len(list(tests_root.glob("campaign_*"))) if tests_root.exists() else 0
        directory = tests_root / f"campaign_{campaign_number}"
        directory.mkdir(parents=True)
        common = yaml.safe_load((repository_root / "configs/generation/common.yaml").read_text(encoding="utf-8"))
        operations = yaml.safe_load((repository_root / "configs/generation/operations/fixed_bed.yaml").read_text(encoding="utf-8"))
        profile = _profile_configuration(simulation_profile, repeated_airflow_times=repeated_airflow_times)
        execution = yaml.safe_load((repository_root / "configs/generation/execution/cluster_cpu.yaml").read_text(encoding="utf-8"))
        execution["runtime"]["timeout_seconds"] = timeout
        execution["retention"]["technical_runtime_smoke"] = {
            "retain_raw_csv": retain_raw_csv,
            "retain_solved_model": retain_solved_model,
        }
        execution["cluster"].update(
            {
                "max_nodes": 2,
                "cases_per_node": 2,
                "cores_per_case": 1,
                "max_parallel_cases": 3,
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

        if material_families != ("lentil",):
            message = "Synthetic runtime fixtures use the canonical one-material technical smoke."
            raise ValueError(message)
        if natural_count != _SMOKE_CASE_COUNT or parameter_ood_count != 0:
            message = "Synthetic runtime fixtures use exactly two natural cases and no parameter OOD."
            raise ValueError(message)
        campaign_seed = 9910 if simulation_profile == "steady_flow" else 9920
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
            "paired_equivalence_seed": 9930,
            "material_roles": {
                "seen": ["lentil"],
                "near_family_ood": [],
                "far_family_ood": [],
                "extreme_family_ood": [],
            },
            "sampling": {
                "method": "lhs",
                "seed_base": campaign_seed,
                "counts": {"natural": {"lentil": 2}},
            },
            "dataset_packages": [
                {"evaluation_regime": "id", "source_role": "seen"},
            ],
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


def scalar_values():
    with pathlib.Path("scalars.csv").open(encoding="utf-8", newline="") as stream:
        return {row["name"]: float(row["value"]) for row in csv.DictReader(stream, delimiter=";")}


mode = os.environ.get("FAKE_COMSOL_MODE", "success")
if mode == "failure":
    print("synthetic failure", file=sys.stderr)
    raise SystemExit(7)
update_tracker(1)
try:
    wait_for_expected_starts()
    if mode == "timeout":
        time.sleep(2.0)
    else:
        time.sleep(float(os.environ.get("FAKE_COMSOL_DELAY", "0")))
        arguments = sys.argv[1:]
        pathlib.Path(arguments[arguments.index("-outputfile") + 1]).write_bytes(b"synthetic solved model\n")
        case = json.loads(pathlib.Path("case.json").read_text(encoding="utf-8"))
        with pathlib.Path("fields.csv").open(encoding="utf-8", newline="") as stream:
            inputs = list(csv.DictReader(stream, delimiter=";"))
        transient_profile = case["simulation_profile"] == "transient_drying"
        scalars = scalar_values() if transient_profile else {}
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
        repeated = os.environ.get("FAKE_COMSOL_REPEAT_AIRFLOW") == "1"
        varying = os.environ.get("FAKE_COMSOL_VARY_AIRFLOW") == "1"
        static_names = ["x", "y", "Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc"]
        if transient_profile:
            static_names.append("X_0_db_field")
        static_names.extend(("u", "v", "p"))
        if transient_profile:
            static_names.append("rho_bu_dry")
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
                    }
                    if transient_profile:
                        values["rho_bu_dry"] = scalars["rho_bu_dry_ref"] * (1.0 - eps) / (1.0 - scalars["eps_bed_cal_ref"])
                    if varying and stationary_time == 1.0 and row_index == 0:
                        values["p"] = 99.0
                    if stationary_time is not None:
                        values["stationary_time"] = stationary_time
                    writer.writerow(values)
        if transient_profile:
            state_times = (0.0, 1.0)
            water_by_time = {}
            with (exports / "transient.csv").open("w", encoding="utf-8", newline="") as stream:
                names = ["x", "y", "t", "T", "phi", "w_surf", "w_int"]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                for state_time in state_times:
                    water_by_time[state_time] = []
                    for source, rho in zip(inputs, rho_values):
                        initial_water = rho * float(source["X_0_db_field"])
                        w_surf = initial_water - 0.2 * state_time
                        w_int = initial_water - 0.1 * state_time
                        water = scalars["f_surf"] * w_surf + (1.0 - scalars["f_surf"]) * w_int
                        water_by_time[state_time].append(water)
                        writer.writerow(
                            {
                                "x": float(source["x"]) + 1e-13,
                                "y": float(source["y"]) + 1e-13,
                                "t": state_time,
                                "T": 296.0 - 0.1 * state_time,
                                "phi": 0.5 - 0.01 * state_time,
                                "w_surf": w_surf,
                                "w_int": w_int,
                            }
                        )
            with (exports / "globals.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "t",
                    "X_wb_bulk",
                    "f_wet_dm",
                    "m_w_gr",
                    "m_v_gas",
                    "m_dot_evap",
                    "m_dot_v_in",
                    "m_dot_v_out",
                    "mt_mass_balance",
                    "T_out_mean",
                    "phi_out_mean",
                ]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                for state_time in state_times:
                    water_values = water_by_time[state_time]
                    weighted_water = sum(water * weight for water, weight in zip(water_values, cell_weights))
                    bulk = weighted_water / (weighted_dry + weighted_water)
                    writer.writerow(
                        dict(
                            zip(
                                names,
                                [
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
                                ],
                            )
                        )
                    )
            with (exports / "status.csv").open("w", encoding="utf-8", newline="") as stream:
                names = [
                    "t_final",
                    "f_wet_dm_final",
                    "X_wb_bulk_final",
                    "X_wb_max_final",
                    "T_min_final",
                    "T_max_final",
                    "phi_min_final",
                    "phi_max_final",
                ]
                writer = csv.DictWriter(stream, fieldnames=names, delimiter=";", lineterminator="\n")
                writer.writeheader()
                final_water = water_by_time[1.0]
                weighted_water = sum(water * weight for water, weight in zip(final_water, cell_weights))
                bulk = weighted_water / (weighted_dry + weighted_water)
                maximum = max(water / (rho + water) for water, rho in zip(final_water, rho_values))
                writer.writerow(
                    dict(
                        zip(
                            names,
                            [1.0, 0.04, bulk, maximum, 295.9, 295.9, 0.49, 0.49],
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
