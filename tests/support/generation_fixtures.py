"""Provide layered synthetic Generation configurations and fake-COMSOL executables.

Responsibilities:
  - Build test-owned campaigns, templates, export mappings, and controllable fake solver processes
  - Expose plain builders so session-scoped fixtures can reuse the same synthetic inputs
  - Register the fake Slurm fixtures that node-mode campaign tests share

Design principles:
  - Generation fixtures isolate Slurm job identity and bind synthetic cases to exact fake commits

This module does NOT:
  - Run licensed COMSOL or validate physical solver accuracy
"""

from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from src import common, generation
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

pytest_plugins = ("support.fake_slurm",)

_DEFAULT_CASE_COUNT = 2
_TEST_MATERIAL_FAMILY = "lentil"
_TEST_STEADY_SEED = 41001
_TEST_TRANSIENT_SEED = 41002
_TEST_GRID = {"nx": 17, "ny": 11}


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
    profile = profiles.resolve_profile(simulation_profile)
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
def _isolate_slurm_job_identity(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Let synthetic tests own job identity instead of inheriting the test allocation."""
    if request.node.get_closest_marker("real_data") is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)


@pytest.fixture(autouse=True)
def _git_commit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind every synthetic case and campaign to one exact fake commit."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", "a" * 40)


_PRODUCTION_YAML: dict[str, Any] = {}


def _production_yaml(relative_path: str) -> Any:
    """Return a private copy of one production Generation config, parsed once per session."""
    if relative_path not in _PRODUCTION_YAML:
        path = Path(__file__).resolve().parents[2] / "configs/01_generation" / relative_path
        _PRODUCTION_YAML[relative_path] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return copy.deepcopy(_PRODUCTION_YAML[relative_path])


def make_generation_config_factory(project_root: Path) -> Callable[..., tuple[Path, Path]]:
    """Return a factory for complete layered synthetic campaigns below one project root.

    The caller owns ``PROJECT_ROOT``; it must point at ``project_root`` while campaigns load.
    """
    repository_root = Path(__file__).resolve().parents[2]
    node_script = project_root / "scripts/generation_node.sh"
    node_script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository_root / "scripts/generation_node.sh", node_script)

    registry = _production_yaml("physical_model/registry.yaml")
    sources = _production_yaml("physical_model/sources.yaml")
    materials_root = project_root / "configs/01_generation/physical_model/materials"
    materials_root.mkdir(parents=True, exist_ok=True)
    for material_family in (_TEST_MATERIAL_FAMILY,):
        material = _production_yaml(f"physical_model/materials/{material_family}.yaml")
        material["oswin"]["record_id"] = "test_owned_lentil_oswin_v1"
        material["oswin"]["components"]["C_osw"] = 0.8
        material["oswin"]["provenance"].update(
            {
                "evidence": "synthetic_design",
                "source_refs": [],
                "method": (
                    "Test-owned non-default sorption record chosen to keep compact "
                    "synthetic startup cases inside their authored operational envelope."
                ),
            }
        )
        (materials_root / f"{material_family}.yaml").write_text(
            yaml.safe_dump(material, sort_keys=False),
            encoding="utf-8",
        )

    catalog_common = _production_yaml("physical_model/common.yaml")
    profiles_root = project_root / "configs/01_generation/physical_model/profiles"
    profiles_root.mkdir(parents=True, exist_ok=True)
    for profile_id in profiles.available_profiles():
        template_relative_path = f"templates/catalog_{profile_id}.mph"
        template_path = project_root / template_relative_path
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_bytes(f"synthetic catalog template for {profile_id}\n".encode())
        template_path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(template_path.read_bytes()).hexdigest()}\n",
            encoding="utf-8",
        )
        profile = _profile_configuration(
            profile_id,
            repeated_airflow_times=False,
            fixed_values=catalog_common["scientific_fixed_values"],
        )
        profile["template"] = template_relative_path
        (profiles_root / f"{profile_id}.yaml").write_text(
            yaml.safe_dump(profile, sort_keys=False),
            encoding="utf-8",
        )

    def build(
        *,
        simulation_profile: str = "transient_drying",
        material_families: tuple[str, ...] = (_TEST_MATERIAL_FAMILY,),
        executable: Path | None = None,
        timeout: float = 5.0,
        scheduler_kind: str = "local",
        retain_solved_model: bool | None = None,
        retain_raw_csv: bool | None = None,
        repeated_airflow_times: bool = False,
        extra_arguments: tuple[str, ...] = (),
        natural_count: int = _DEFAULT_CASE_COUNT,
        parameter_ood_count: int = 0,
        startup_ramp_enabled: bool = True,
        startup_ramp_duration_h: float = 0.25,
        max_admission_cases: int = 1,
        poll_interval_seconds: int = 1,
        node_jobs: int = 2,
        cores_per_case: int = 1,
        cores_per_node: int = 24,
        node_job_wall_time: str | None = None,
        license_retry_enabled: bool = True,
        license_initial_delay_seconds: float = 2.0,
        license_maximum_delay_seconds: float = 5.0,
        license_maximum_wait_seconds: float = 12.0,
        in_allocation_retry_enabled: bool = True,
        in_allocation_maximum_window_seconds: float = 0.08,
        in_allocation_pause_after_capacity_failure_seconds: float = 0.01,
        grid_overrides: Mapping[str, int | float] | None = None,
        campaign_purpose: str = "source_generation",
    ) -> tuple[Path, Path]:
        tests_root = project_root / "configs/01_generation/test_support"
        campaign_number = len(list(tests_root.glob("campaign_*"))) if tests_root.exists() else 0
        directory = tests_root / f"campaign_{campaign_number}"
        directory.mkdir(parents=True)
        common = _production_yaml("physical_model/common.yaml")
        common["grid"] = {
            **common["grid"],
            **_TEST_GRID,
            **({} if grid_overrides is None else dict(grid_overrides)),
        }
        operations = _production_yaml("physical_model/fixed_bed.yaml")
        operations["boundary_schedule"]["startup_ramp"] = {
            "enabled": startup_ramp_enabled,
            "duration_h": startup_ramp_duration_h,
            "initial_equilibrium_rh_dry_margin": 0.05,
            "max_relative_humidity": 0.90,
        }
        template_relative_path = f"templates/{simulation_profile}_{campaign_number}.mph"
        template_path = project_root / template_relative_path
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_bytes(f"synthetic COMSOL template for {simulation_profile} #{campaign_number}\n".encode())
        template_path.with_suffix(".sha256").write_text(
            f"{hashlib.sha256(template_path.read_bytes()).hexdigest()}\n",
            encoding="utf-8",
        )
        profile = _profile_configuration(
            simulation_profile,
            repeated_airflow_times=repeated_airflow_times,
            fixed_values=common["scientific_fixed_values"],
        )
        profile["template"] = template_relative_path
        execution = _production_yaml("execution.yaml")
        execution["runtime"].update(
            {
                "timeout_seconds": timeout,
                "graceful_stop_reserve_seconds": float(timeout) / 2.0 if float(timeout) > 0.0 else 1.0,
                "extra_arguments": list(extra_arguments),
                "license_wait": {
                    "enabled": license_retry_enabled,
                    "initial_delay_seconds": license_initial_delay_seconds,
                    "maximum_delay_seconds": license_maximum_delay_seconds,
                    "maximum_wait_seconds": license_maximum_wait_seconds,
                    "in_allocation_retry": {
                        "enabled": in_allocation_retry_enabled,
                        "maximum_window_seconds": in_allocation_maximum_window_seconds,
                        "pause_after_capacity_failure_seconds": in_allocation_pause_after_capacity_failure_seconds,
                    },
                },
            }
        )
        if retain_raw_csv is None and retain_solved_model is None:
            retention_policy = "compact"
        else:
            retention_policy = "full" if retain_raw_csv or retain_solved_model else "compact"
        execution["retention"] = retention_policy
        execution["submission"].update(
            {
                "max_admission_cases": max_admission_cases,
                "poll_interval_seconds": poll_interval_seconds,
            }
        )
        # One case plus the 600 s publication margin and a test-owned reserve must fit into a fresh node job.
        wall_seconds = max(1, int(float(timeout))) + 900
        derived_wall_time = f"{wall_seconds // 3600:02d}:{(wall_seconds % 3600) // 60:02d}:{wall_seconds % 60:02d}"
        execution["cluster"].update(
            {
                "cores_per_case": cores_per_case,
                "node_jobs": node_jobs,
                "memory_per_case": "1G",
                "node_memory_overhead": "1G",
                "node_memory_limit": "100G",
                "node_job_wall_time": derived_wall_time if node_job_wall_time is None else node_job_wall_time,
                "scheduler_options": [],
            }
        )
        execution["site"].update(
            {
                "cpu_host": "synthetic-cpu.example",
                "scheduler": scheduler_kind,
                "partition": "test",
                "cores_per_node": cores_per_node,
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
        if campaign_purpose != "source_generation":
            message = "Synthetic fixtures support ordinary source generation."
            raise ValueError(message)
        campaign_seed = _TEST_STEADY_SEED if simulation_profile == "steady_flow" else _TEST_TRANSIENT_SEED
        campaign_name = f"lentil-{natural_count}__ref-range__lhs-seed-{campaign_seed}"
        campaign = {
            "schema_kind": "generation_source_campaign",
            "schema_version": 1,
            "sources_config": "sources.yaml",
            "registry_config": "registry.yaml",
            "common_config": "common.yaml",
            "operations_config": "operations.yaml",
            "profile_config": "profile.yaml",
            "execution_config": "execution.yaml",
            "materials": [_TEST_MATERIAL_FAMILY],
            "domain_profile": "reference_range",
            "sampling": {
                "method": "lhs",
                "seed_base": campaign_seed,
                "counts": {"natural": {_TEST_MATERIAL_FAMILY: natural_count}},
            },
        }
        config_path = directory / f"{campaign_name}.yaml"
        config_path.write_text(yaml.safe_dump(campaign, sort_keys=False), encoding="utf-8")
        return config_path, template_path

    return build


@pytest.fixture
def generation_config_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[Path, Path]]:
    """Return a per-test campaign factory whose project root is the active ``PROJECT_ROOT``."""
    project_root = tmp_path / "project"
    monkeypatch.setenv("PROJECT_ROOT", str(project_root))
    return make_generation_config_factory(project_root)


def lentil_generation_config(
    generation_config_factory: Callable[..., tuple[Path, Path]],
    profile_id: str,
    *,
    storage_root: Path | None = None,
    startup_enabled: bool = True,
    startup_duration_h: float = 0.25,
    natural_count: int = 4,
) -> Any:
    """Build one compact natural-lentil batch configuration, optionally registering its campaign."""
    path, _template = generation_config_factory(
        simulation_profile=profile_id,
        natural_count=natural_count,
        startup_ramp_enabled=startup_enabled,
        startup_ramp_duration_h=startup_duration_h,
    )
    if storage_root is not None:
        generation.lifecycle.create_campaign(generation.cases.config.load_campaign_config(path), storage_root=storage_root)
    return generation.cases.config.load_generation_config(
        path,
        only_batch=generation.cases.config.build_batch_name(profile_id, _TEST_MATERIAL_FAMILY, "natural"),
    )


def successful_source_case(campaign_name: str, batch: Any, case_index: int, storage: Path, work_root: Path) -> dict[str, Any]:
    """Run, validate, and decide one exact fake-boundary Source candidate; return its provenance."""
    outcome = generation.runtime.run_case(batch, case_index, cores_per_case=1, storage_root=storage, work_root=work_root)
    assert outcome.status == "completed"
    provenance = generation.runtime.validate_completed_case(batch, case_index, storage_root=storage, validation_depth="routine")
    artifact = outcome.processed_directory / "case.h5"
    generation.lifecycle.decide_candidate(
        batch.profile.id,
        campaign_name,
        batch.material_family,
        outcome.case_id,
        outcome="successful",
        stage="publication",
        failure_class=None,
        reason_code=None,
        detail="Validated synthetic Source publication",
        case_input_id=provenance["case_input_id"],
        simulation_case_id=provenance["simulation_case_id"],
        case_hdf5_relative=artifact.relative_to(storage).as_posix(),
        case_hdf5_sha256=common.serialization.file_sha256(artifact),
        evidence={},
        storage_root=storage,
    )
    return provenance


@pytest.fixture
def fake_comsol(tmp_path: Path) -> Path:
    """Return an executable emitting complete synthetic outputs for both profiles."""
    return write_fake_comsol(tmp_path)


def write_fake_comsol(directory: Path) -> Path:
    """Write the fake COMSOL executable into ``directory`` and return its path."""
    path = directory / "fake_comsol.py"
    path.write_text(
        r"""#!/usr/bin/env python3
import csv
import fcntl
import json
import math
import os
import pathlib
import signal
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


def write_capacity_status():
    state = os.environ.get("FAKE_COMSOL_CAPACITY_STATUS_STATE")
    if state is None:
        return
    content = f"1787215759123\n{state}\n"
    pathlib.Path("solved.mph.status").write_text(content, encoding="utf-8")


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
    fixed_names = ["Nx", "Ny", "Lx", "Ly", "Lz", "dt_out", "t_max"]
    expected_names = [entry["name"] for entry in runtime_entries] + fixed_names
    expected_indices = [str(index) for index in range(1, len(expected_names) + 1)]
    if names != expected_names or indices != expected_indices or len(values) != len(expected_names):
        raise RuntimeError("fake COMSOL received a misordered or incomplete runtime scalar vector")
    parsed = {}
    for encoded, entry in zip(values[:12], runtime_entries):
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
    fixed = dict(zip(fixed_names, values[12:]))
    for name in ("Nx", "Ny"):
        if not fixed[name].isdigit() or int(fixed[name]) < 2:
            raise RuntimeError(f"fake COMSOL grid parameter {name} is invalid")
    for name, unit in (("Lx", "m"), ("Ly", "m"), ("Lz", "m"), ("dt_out", "h"), ("t_max", "h")):
        suffix = f"[{unit}]"
        if not fixed[name].endswith(suffix):
            raise RuntimeError(f"fake COMSOL discretization parameter {name} lacks {unit}")
        number_text = fixed[name][: -len(suffix)]
        number = float(number_text)
        if not math.isfinite(number) or number <= 0 or format(number, ".17g") != number_text:
            raise RuntimeError(f"fake COMSOL discretization parameter {name} is not canonical positive text")
    with pathlib.Path("fields.csv").open(encoding="utf-8", newline="") as stream:
        spatial = list(csv.DictReader(stream, delimiter=";"))
    if len({row["x"] for row in spatial}) != int(fixed["Nx"]) or len({row["y"] for row in spatial}) != int(fixed["Ny"]):
        raise RuntimeError("fake COMSOL grid controls disagree with the canonical spatial input")
    with pathlib.Path("schedule.csv").open(encoding="utf-8", newline="") as stream:
        schedule = list(csv.DictReader(stream, delimiter=";"))
    times = [float(row["t"]) for row in schedule]
    if not times or times[0] != 0 or times[-1] < float(fixed["t_max"][:-3]) - 1e-9:
        raise RuntimeError("fake COMSOL end time exceeds the canonical schedule")
    if len(times) > 1 and not any(abs((right - left) - float(fixed["dt_out"][:-3])) <= 1e-9 for left, right in zip(times, times[1:])):
        raise RuntimeError("fake COMSOL output interval disagrees with the canonical schedule")
    return parsed


if sys.argv[1:] == ["-version"]:
    print("COMSOL Multiphysics 6.4.0.293")
    raise SystemExit(0)

arguments = sys.argv[1:]
if arguments.count("-batchlog") != 1 or arguments.count("-batchlogout") != 1:
    raise RuntimeError("fake COMSOL requires one complete runtime-owned batch-log flag pair")
batch_log = pathlib.Path(arguments[arguments.index("-batchlog") + 1])


def write_comsol_batch_log(content):
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    with batch_log.open("w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


mode = os.environ.get("FAKE_COMSOL_MODE", "success")
if mode == "license_capacity_twice_then_success":
    counter_path = pathlib.Path("runtime/fake_license_checkout_count")
    previous = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
    counter_path.write_text(str(previous + 1), encoding="utf-8")
    mode = "license_capacity" if previous < 2 else "success_after_capacity"
if mode == "failure":
    print("Step        Time    Stepsize      Res  Jac  Sol Order Tfail NLfail   LinErr   LinRes", flush=True)
    print("   1      1.6355     0.42589       18    6   18     1     0      0  4.8e-16  4.6e-16", flush=True)
    print("synthetic failure", file=sys.stderr)
    raise SystemExit(7)
if mode == "external_sigterm":
    os.kill(os.getpid(), 15)
    time.sleep(1.0)
    raise RuntimeError("fake COMSOL survived SIGTERM")
if mode == "silent_startup":
    time.sleep(30.0)
    raise RuntimeError("fake COMSOL exceeded its controller-owned startup window")
if mode == "license_capacity_delayed":
    tracker_registered = False

    def unregister_delayed_checkout():
        global tracker_registered
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            if tracker_registered:
                update_tracker(-1)
                tracker_registered = False
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def finish_delayed_checkout_on_term(signum, _frame):
        # Also finalize a signal arriving before the normal finalizer can mask it.
        unregister_delayed_checkout()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, finish_delayed_checkout_on_term)
    # Protect registration until its finalizer is installed; keep checkout unmasked.
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        update_tracker(1)
        tracker_registered = True
        try:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            wait_for_expected_starts()
            print("Could not obtain license for 'Brinkman Equations (br)'.")
            print("License error: -4.", file=sys.stderr)
            print("Licensed number of users already reached.", file=sys.stderr)
            time.sleep(0.1)
        finally:
            unregister_delayed_checkout()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    raise SystemExit(0)
if mode in {"license_capacity", "success_with_license_warning"}:
    print("Could not obtain license for 'Brinkman Equations (br)'.")
    print("Required product: CFD Module.")
    print("License error: -4.", file=sys.stderr)
    print("Licensed number of users already reached.", file=sys.stderr)
    print("Feature: COMSOL", file=sys.stderr)
    print("FlexNet Licensing error:-4,132", file=sys.stderr)
    if mode == "license_capacity":
        write_capacity_status()
        raise SystemExit(0)
if mode == "success_after_capacity":
    print("Time-Dependent Solver 1 in Transient Drying", flush=True)
update_tracker(1)
try:
    wait_for_expected_starts()
    if mode == "timeout":
        time.sleep(2.0)
    else:
        time.sleep(float(os.environ.get("FAKE_COMSOL_DELAY", "0")))
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
        stationary_batch_log = (
            "<---- Dependent Variables 1 in Stationary Airflow/Stationary Airflow Solution\n"
            "      (sol1) -------------------------------------------------------------------\n"
            "Solution time: 0.5 s.\n"
            "----- Dependent Variables 1 in Stationary Airflow/Stationary Airflow Solution\n"
            "      (sol1) ------------------------------------------------------------------>\n"
            "<---- Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution\n"
            "      (sol1) -------------------------------------------------------------------\n"
            "Solution time: 3.5 s.\n"
            "----- Stationary Solver 1 in Stationary Airflow/Stationary Airflow Solution\n"
            "      (sol1) ------------------------------------------------------------------>\n"
        )
        transient_batch_log = (
            "<---- Dependent Variables 1 in Transient Drying/Transient Drying Solution (sol2) -\n"
            "Values of variables not solved for: Stationary Airflow Solution (sol1).\n"
            "Solution time: 0.75 s.\n"
            "----- Dependent Variables 1 in Transient Drying/Transient Drying Solution (sol2) >\n"
            "<---- Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution\n"
            "      (sol2) -------------------------------------------------------------------\n"
            "Time-dependent solver (BDF)\n"
            "Solution time: 7.25 s.\n"
            "----- Time-Dependent Solver 1 in Transient Drying/Transient Drying Solution\n"
            "      (sol2) ------------------------------------------------------------------>\n"
        )
        if mode == "success_missing_transient_timing":
            write_comsol_batch_log(stationary_batch_log)
        else:
            write_comsol_batch_log(stationary_batch_log + (transient_batch_log if transient_profile else ""))
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
                scalars["rho_bu_dry_ref"] * (1.0 - float(source["eps_bed"])) / (1.0 - scalars["eps_bed_ref"])
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
                        values["rho_bu_dry"] = scalars["rho_bu_dry_ref"] * (1.0 - eps) / (1.0 - scalars["eps_bed_ref"])
                    if varying and stationary_time == 1.0 and row_index == 0:
                        values["p"] = 99.0
                    if stationary_time is not None:
                        values["stationary_time"] = stationary_time
                    writer.writerow(values)
        if mapping_export_mode == "missing":
            airflow_path.unlink()
        if transient_profile:
            transient_time_mode = os.environ.get("FAKE_COMSOL_TRANSIENT_TIME_MODE", "exact_stop")
            if transient_time_mode == "exact_stop":
                state_times = (0.0, 1.0, 1.5)
            elif transient_time_mode == "short_rollout":
                state_times = (0.0, 1.0, 2.0, 2.5)
            elif transient_time_mode == "configured_horizon":
                parameter_names = sys.argv[sys.argv.index("-pname") + 1].split(",")
                parameter_values = sys.argv[sys.argv.index("-plist") + 1].split(",")
                controls = dict(zip(parameter_names, parameter_values))
                stop = float(controls["t_max"][:-3])
                interval = float(controls["dt_out"][:-3])
                state_times = tuple(index * interval for index in range(round(stop / interval) + 1))
            elif transient_time_mode == "post_horizon":
                state_times = (*map(float, range(169)), 172.2885558527809)
            else:
                raise RuntimeError(f"unsupported fake transient time mode: {transient_time_mode}")
            water_by_time = {state_time: [] for state_time in state_times}
            f_wet_by_time = {
                state_time: (
                    1.0
                    if state_time == 0.0
                    else 0.04
                    if (
                        transient_time_mode == "exact_stop"
                        or (transient_time_mode == "short_rollout" and state_time == state_times[-1])
                        or (transient_time_mode == "post_horizon" and state_time > 168.0)
                    )
                    else 0.08
                )
                for state_time in state_times
            }
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
                            f_wet_by_time[state_time],
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
                        f_wet_by_time[final_time],
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
