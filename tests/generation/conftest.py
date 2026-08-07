"""Synthetic configuration and fake-COMSOL fixtures for generation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src import generation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def generation_config_factory(tmp_path: Path) -> Any:
    """Return a factory for small explicit profile-qualified configurations."""

    def build(
        *,
        simulation_profile: str = "transient_drying",
        case_indices: list[int] | None = None,
        executable: Path | None = None,
        timeout: float = 2.0,
        scheduler_kind: str = "local",
        scalar_entries: list[dict[str, Any]] | None = None,
        retain_solved_model: bool = False,
        repeated_airflow_times: bool = False,
    ) -> tuple[Path, Path]:
        profile = generation.profiles.get_profile(simulation_profile)
        transient = simulation_profile == generation.profiles.TRANSIENT_DRYING_PROFILE
        steady_export: dict[str, Any] = {
            "role": "steady_flow_fields",
            "pattern": "airflow.csv",
            "required": True,
            "allow_multiple": False,
            "format": "numeric_table",
            "delimiter": ";",
            "columns": {"x": "x", "y": "y", "p": "p", "u": "u", "v": "v"},
        }
        if repeated_airflow_times:
            steady_export["time_column"] = "time"
        contracts = [steady_export]
        if transient:
            contracts.append(
                {
                    "role": "transient_fields",
                    "pattern": "transient_*.csv",
                    "required": True,
                    "allow_multiple": True,
                    "format": "numeric_table",
                    "delimiter": ";",
                }
            )
        inputs: dict[str, Any] = {
            "spatial_files": [
                {
                    "filename": "case_0001.csv",
                    "delimiter": ";",
                    "columns": ["x", "y", "Kxx", "Kxy", "Kyy", "eps", "p_bc"],
                }
            ],
        }
        if transient:
            inputs.update(
                {
                    "scalar_file": {
                        "filename": "scalar_values.csv",
                        "format": "long",
                        "delimiter": ";",
                        "include_header": True,
                        "required_when_empty": False,
                        "entries": scalar_entries
                        if scalar_entries is not None
                        else [
                            {"name": "alpha", "value": 2.5, "unit": "1/s"},
                            {"name": "beta", "value": 4.0},
                        ],
                    },
                    "schedule_file": {
                        "filename": "schedule.csv",
                        "delimiter": ";",
                        "columns": ["time", "control"],
                        "rows": [[0.0, 1.0], [5.0, 2.0]],
                    },
                }
            )
        values: dict[str, Any] = {
            "schema_version": 1,
            "simulation_profile": simulation_profile,
            "cases": {"indices": case_indices or [1], "seed_base": 3001, "overrides": {}},
            "generator": {
                "version": "python_multiscale_v1",
                "domain": {"length_x_m": 0.04, "length_y_m": 0.03, "resolution_m": 0.01},
                "parameters": {
                    "base_len_rel": 0.20,
                    "smooth_len_rel": 0.10,
                    "ms_weight": [0.3, 0.7],
                    "anisotropy": [2.0, 1.0],
                    "coupling": 0.5,
                    "noise_level": 0.0,
                    "noise_granularity": 0.5,
                    "noise_bias": 0.5,
                    "k_mean": 5e-9,
                    "var_rel": 0.5,
                    "a_max": 2.0,
                    "a_gamma": 2.0,
                    "tensor_strength": 1.0,
                    "theta_jitter": 0.0,
                    "theta_smooth_rel": 0.1,
                    "A_rel": 2.0,
                    "eps_min_global": 0.3,
                    "eps_max_global": 0.8,
                    "eps_smooth_rel": 0.05,
                    "texture_amp": 0.005,
                    "p_inlet_mean": 350.0,
                    "a_sin": 0.03,
                    "f_sin": 0.75,
                    "phi_sin": 3.141592653589793,
                    "k_gauss": 2,
                    "a_gauss": 0.05,
                    "sigma_gauss": 0.05,
                    "gauss_jitter": 0.25,
                    "a_lin": 0.025,
                },
            },
            "inputs": inputs,
            "exports": {"root": "exports", "contracts": contracts},
            "execution": {
                "executable": None if executable is None else str(executable),
                "timeout_seconds": timeout,
                "retain_solved_model": retain_solved_model,
                "extra_arguments": [],
            },
            "cluster": {"cores_per_node": 32, "scheduler_kind": scheduler_kind, "scheduler_options": []},
        }
        config_path = tmp_path / f"generation_{len(list(tmp_path.glob('generation_*.yaml')))}.yaml"
        config_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
        return config_path, profile.template_path

    return build


@pytest.fixture
def fake_comsol(tmp_path: Path) -> Path:
    """Return an executable emulating both profile outputs and process outcomes."""
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
        output = pathlib.Path(arguments[arguments.index("-outputfile") + 1])
        output.write_bytes(b"synthetic solved model\n")
        case = json.loads(pathlib.Path("case.json").read_text(encoding="utf-8"))
        with pathlib.Path("case_0001.csv").open(encoding="utf-8", newline="") as stream:
            inputs = list(csv.DictReader(stream, delimiter=";"))
        exports = pathlib.Path("exports")
        exports.mkdir(exist_ok=True)
        repeated = os.environ.get("FAKE_COMSOL_REPEAT_AIRFLOW") == "1"
        varying = os.environ.get("FAKE_COMSOL_VARY_AIRFLOW") == "1"
        header = "time;x;y;p;u;v\n" if repeated else "x;y;p;u;v\n"
        rows = [header]
        times = (0.0, 1.0) if repeated else (None,)
        for time_value in times:
            for row_index, values in enumerate(inputs):
                airflow = [values["x"], values["y"], str(10.0 + row_index), str(0.1 + row_index), str(0.2 + row_index)]
                if varying and time_value == 1.0 and row_index == 0:
                    airflow[2] = "99"
                if time_value is not None:
                    airflow.insert(0, str(time_value))
                rows.append(";".join(airflow) + "\n")
        (exports / "airflow.csv").write_text("".join(rows), encoding="utf-8")
        if case["simulation_profile"] == "transient_drying":
            (exports / "transient_000.csv").write_text("time;x;y;T;moisture\n0;0;0;300;0.2\n", encoding="utf-8")
        print("synthetic success")
finally:
    update_tracker(-1)
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path
