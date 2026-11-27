"""Build synthetic field-test directories with Source-format case files for the field-test analysis tests.

Responsibilities:
  - Write compact ``case.h5`` files with the Generation source-case members the field-test reader and gate comparison use
  - Write a published base Source case, a frozen roster (through the Generation roster authority), member seals, attempt
    outcomes and member cases in the interface layout, below the resolved field-test directory
  - Freeze stage A's tolerance table for a stage-B field test and bind it through the roster's ``derived_from``
  - Build validation pairs whose prediction error pattern sets known tolerances

Design principles:
  - Fields are small (64 x 96 cells) but keep the real names, units, groups and JSON provenance encoding
  - Every variant is the base plus a scaled copy of one error pattern, so ratios against the tolerance are predictable

This module does NOT:
  - Run Generation, a solver or a trained model
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from src import common, generation
from src.analysis import field_tests
from src.generation.contracts import generation_contracts_profiles as profiles

if TYPE_CHECKING:
    from pathlib import Path

GRID = (64, 96)
HOURS = np.arange(5, dtype=np.float64)
CAMPAIGN = "fixture-source"
TEST_ID = "fixture-test"
STAGE_A_TEST_ID = "fixture-stage-a"
_STRING = h5py.string_dtype(encoding="utf-8")


def _named(group: h5py.Group, name: str, values: np.ndarray, names: tuple[str, ...]) -> None:
    """Write one named-field dataset with its JSON field names."""
    dataset = group.create_dataset(name, data=values)
    dataset.attrs["field_names"] = json.dumps(list(names))


def steady_fields(seed: int) -> dict[str, np.ndarray]:
    """Return smooth positive steady fields of one base."""
    y, x = np.meshgrid(np.linspace(0.0, 1.0, GRID[0]), np.linspace(0.0, 1.0, GRID[1]), indexing="ij")
    scale = 1.0 + 0.1 * seed
    return {
        "Kxx": 1.0e-8 * scale * (1.0 + x),
        "Kxy": 1.0e-10 * scale * x * y,
        "Kyy": 1.0e-8 * scale * (1.0 + y),
        "eps_bed": 0.4 + 0.01 * seed + 0.0 * x,
        "p_in_bc": 100.0 * scale * (1.0 - y),
        "u": 0.01 * scale * np.sin(np.pi * x),
        "v": 0.1 * scale * (1.0 + 0.2 * x),
        "p": 100.0 * scale * (1.0 - y),
    }


def error_pattern(shape: tuple[int, ...]) -> np.ndarray:
    """Return a positive convex profile across the walls, so every patch, norm and contrast endpoint sees it."""
    x = np.linspace(0.0, 1.0, shape[-1])
    return np.broadcast_to(1.0 + x**2, shape).copy()


def transient_states(seed: int) -> np.ndarray:
    """Return drying-like states ``[5, 4, Y, X]`` of one base."""
    time = HOURS[:, None, None]
    y = np.linspace(0.0, 1.0, GRID[0])[None, :, None]
    base = np.zeros((len(HOURS), 4, *GRID))
    base[:, 0] = 300.0 + 2.0 * time * (1.0 - 0.5 * y) + seed
    base[:, 1] = 0.6 - 0.05 * time * (1.0 - 0.5 * y)
    base[:, 2] = 200.0 - 10.0 * time * (1.0 - 0.5 * y)
    base[:, 3] = 220.0 - 5.0 * time * (1.0 - 0.5 * y)
    return base


def write_case_h5(
    path: Path,
    *,
    task: str,
    fields: dict[str, np.ndarray] | None = None,
    times: np.ndarray | None = None,
    states: np.ndarray | None = None,
    t_final: float = 3.5,
    wet_final: float = 0.01,
    horizon: float = 10.0,
    wet_fractions: np.ndarray | None = None,
) -> Path:
    """Write one Source-format ``case.h5`` of either task and return its path; ``wet_fractions`` sets f_wet_dm per stored hour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_kind"] = "generation_source_case_hdf5"
        handle.attrs["simulation_profile"] = task
        coords = handle.create_group("coords")
        coords.create_dataset("x", data=np.linspace(0.0, 1.2, GRID[1]))
        coords.create_dataset("y", data=np.linspace(0.0, 0.75, GRID[0]))
        static_names = profiles.static_field_names(task)
        values = fields if fields is not None else steady_fields(0)
        static = np.stack([values.get(name, np.ones(GRID)) for name in static_names]).astype(np.float32)
        _named(handle.create_group("static"), "fields", static, static_names)
        config: dict[str, Any] = {"scientific_fixed_values": {"f_wet_dm_max": 0.05}, "time": {"stop": horizon}}
        provenance = handle.create_group("provenance")
        provenance.create_dataset("scientific_config_json", data=json.dumps(config), dtype=_STRING)
        if task == profiles.TRANSIENT_DRYING_PROFILE:
            if times is None or states is None:
                msg = "A transient fixture case needs times and states."
                raise ValueError(msg)
            # The writer's hour classification tolerance: 16 eps times the largest time magnitude.
            handle.create_dataset("time", data=times).attrs["classification_atol"] = 16.0 * float(np.finfo(np.float64).eps) * max(1.0, horizon)
            _named(handle.create_group("transient"), "fields", states.astype(np.float32), profiles.TRANSIENT_FIELD_NAMES)
            final = np.array([t_final, wet_final, 0.1, 0.12, 290.0, 310.0, 0.1, 0.7])
            _named(handle.create_group("final_status"), "values", final, profiles.FINAL_STATUS_FIELDS)
            # Global diagnostics hold the stored hours, then the exact stop when it lies after the last stored hour.
            complete = np.append(times, t_final) if t_final > times[-1] else np.asarray(times)
            per_hour = np.linspace(0.5, wet_final, len(times)) if wet_fractions is None else wet_fractions
            diagnostics = np.zeros((complete.size, len(profiles.GLOBAL_FIELD_NAMES)))
            diagnostics[:, profiles.GLOBAL_FIELD_NAMES.index("t")] = complete
            diagnostics[:, profiles.GLOBAL_FIELD_NAMES.index("f_wet_dm")] = np.append(per_hour, wet_final)[: complete.size]
            _named(handle.create_group("global"), "values", diagnostics, profiles.GLOBAL_FIELD_NAMES)
            scalar_names = ("T_amb", "X_target_wb", "f_surf")
            _named(handle.create_group("scalar"), "values", np.array([293.15, 0.12, 0.5]), scalar_names)
    return path


def case_kwargs(task: str, seed: int, *, scale: float = 0.0, **completion: float) -> dict[str, Any]:
    """Return the ``write_case_h5`` arguments of base ``seed`` plus ``scale`` times the error pattern."""
    if task == profiles.STEADY_FLOW_PROFILE:
        fields = steady_fields(seed)
        for name, size in (("p", 1.0), ("u", 1.0e-3), ("v", 1.0e-3)):
            fields[name] = fields[name] + scale * size * error_pattern(GRID)
        return {"task": task, "fields": fields}
    states = transient_states(seed)
    step = np.array([0.1, 0.002, 0.5, 0.5])[None, :, None, None]
    states = states + scale * step * error_pattern(states.shape) * (1.0 + HOURS[:, None, None, None])
    return {"task": task, "times": HOURS.copy(), "states": states, **completion}


def run_identity(run: str = "demo-run") -> dict[str, str]:
    """Return the admitted identity of a fixture run below ``03_experiments``."""
    return {"path": f"03_experiments/{run}", "effective_config_digest": "0" * 64, "best_checkpoint_sha256": "1" * 64}


def validation_pairs(task: str, *, duration_error: float = 2.0, run: str = "demo-run") -> list[field_tests.tolerances.ValidationPair]:
    """Return two validation pairs of the fixture run ``run`` whose prediction error is exactly the unit error pattern."""
    pairs = []
    for seed in (0, 1):
        reference = _fields(task, case_kwargs(task, seed))
        prediction = _fields(task, case_kwargs(task, seed, scale=1.0, t_final=3.5 + duration_error))
        pairs.append(field_tests.tolerances.ValidationPair(run_identity(run), f"case_{seed:04d}", reference, prediction))
    return pairs


def _fields(task: str, kwargs: dict[str, Any]) -> field_tests.endpoints.CaseFields:
    """Return the in-memory case fields the reader would return for ``write_case_h5(**kwargs)``."""
    if task == profiles.STEADY_FLOW_PROFILE:
        values = {name: np.asarray(value, dtype=np.float32).astype(np.float64) for name, value in kwargs["fields"].items()}
        values["speed"] = np.hypot(values["u"], values["v"])
        return field_tests.endpoints.CaseFields(task=task, fields=values)
    completion = field_tests.endpoints.Completion(
        t_stop=float(kwargs.get("t_final", 3.5)),
        reached=True,
        wet_fraction_final=0.01,
        horizon=10.0,
        fraction_limit=0.05,
        target_wet_basis=0.12,
    )
    return field_tests.endpoints.CaseFields(
        task=task, times=kwargs["times"], states=np.asarray(kwargs["states"], dtype=np.float32), completion=completion
    )


def freeze(test_dir: Path, task: str) -> str:
    """Freeze the fixture tolerance table of ``task`` into the field test and return its digest."""
    table = field_tests.tolerances.build_tolerance_table(task, validation_pairs(task))
    return field_tests.tolerances.freeze_tolerances(test_dir / "analysis" / "tolerances.json", table)


def _variant(challenge: dict[str, Any], number: int, reference: dict[str, Any]) -> dict[str, Any]:
    """Return the roster entry of one challenge, or of one joint member paired with its base's gate."""
    index = challenge["base"]
    joint = challenge.get("role") == "joint"
    return {
        "member_id": f"c{number}-{index}",
        "role": challenge.get("role", "challenge"),
        "base": reference["base"],
        "gate_member_id": f"gate-{index}",
        "reference_member_id": f"gate-{index}" if joint else f"ref-{index}",
        "row": None if joint else challenge["row"],
        "level": "joint" if joint else challenge["level"],
        "intervention": challenge.get("intervention", {str(challenge.get("row")): number}),
    }


def make_field_test(
    storage_root: Path,
    *,
    task: str,
    stage: str,
    bases: list[tuple[str, int]],
    challenges: list[dict[str, Any]],
    base_completion: dict[int, dict[str, float]] | None = None,
    gate_cases: dict[int, dict[str, Any]] | None = None,
    published_cases: dict[int, dict[str, Any]] | None = None,
) -> Path:
    """
    Write the field test ``TEST_ID`` with one gate and one nominal reference per base and the given challenges.

    Each challenge mapping holds ``row``, ``level``, ``base`` (index into ``bases``), ``case`` (``write_case_h5``
    arguments or ``None``), optional ``seal`` overrides, ``role`` (``joint`` pairs the member with its base's gate),
    ``intervention`` and ``outcomes`` (the attempt classifications; by default the seal's disposition once per attempt).
    ``base_completion`` overrides the transient completion arguments of a base's published case, gate and reference,
    ``published_cases`` replaces a base's published case and with it its gate's default case, and ``gate_cases``
    replaces a base's gate case. Every member is sealed on CPU model ``fixture`` unless its seal overrides say otherwise.
    A stage-B field test binds stage A's table, frozen here in ``STAGE_A_TEST_ID``.
    """
    completion = base_completion or {}
    test_dir = common.paths.resolve_field_test_dir(task, TEST_ID, storage_root=storage_root)
    source = storage_root / "01_generation" / task / CAMPAIGN
    members: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any] | None] = {}
    specs: dict[str, dict[str, Any]] = {}
    for index, (material, design_index) in enumerate(bases):
        case_path = f"materials/{material}/case_{index + 1:04d}/result/case.h5"
        published_case = (published_cases or {}).get(index, case_kwargs(task, index, **completion.get(index, {})))
        published = write_case_h5(source / case_path, **published_case)
        base = {
            "material_family": material,
            "design_kind": "sobol",
            "design_index": design_index,
            "case_id": f"case_{index + 1:04d}",
            "case_path": case_path,
            "case_sha256": common.serialization.file_sha256(published),
            "case_json_sha256": "0" * 64,
        }
        for role, prefix in (("gate", "gate"), ("reference", "ref")):
            member_id = f"{prefix}-{index}"
            members.append(
                {
                    "member_id": member_id,
                    "role": role,
                    "base": base,
                    "gate_member_id": None if role == "gate" else f"gate-{index}",
                    "reference_member_id": None,
                    "row": None,
                    "level": "nominal",
                    "intervention": {} if role == "gate" else {"fixture.nominal": 0.0},
                    **({"reference_kind": "R0"} if role == "reference" else {}),
                }
            )
            cases[member_id] = case_kwargs(task, index, **completion.get(index, {}))
        cases[f"gate-{index}"] = (gate_cases or {}).get(index, published_case)
    for number, challenge in enumerate(challenges):
        member = _variant(challenge, number, next(item for item in members if item["member_id"] == f"ref-{challenge['base']}"))
        members.append(member)
        cases[member["member_id"]] = challenge["case"]
        specs[member["member_id"]] = challenge
    derived_from = None
    if stage == "B":
        stage_a = common.paths.resolve_field_test_dir(task, STAGE_A_TEST_ID, storage_root=storage_root)
        derived_from = {
            "test_id": STAGE_A_TEST_ID,
            "roster_sha256": "0" * 64,
            "roster_version": 1,
            "verdicts_sha256": "0" * 64,
            "tolerances_sha256": freeze(stage_a, task),
        }
    roster = generation.field_tests.roster.build_roster(
        test_id=TEST_ID,
        task=task,
        stage=stage,
        base_source={"campaign_name": CAMPAIGN, "plan_sha256": "0" * 64},
        members=members,
        max_attempts=3,
        derived_from=derived_from,
    )
    generation.field_tests.roster.write_roster(test_dir, roster)
    record = generation.field_tests.record
    for member in members:
        member_id = member["member_id"]
        member_dir = test_dir / "members" / member_id
        kwargs = cases[member_id]
        spec = specs.get(member_id, {})
        seal: dict[str, Any] = {
            "schema_kind": record.SEAL_SCHEMA_KIND,
            "member_id": member_id,
            "disposition": "succeeded" if kwargs is not None else "solver_failure",
            "attempts": 1,
            "bundle_sha256": "0" * 64,
            "spatial_support_attempt": 0,
            "schedule_acceptance_attempt": None if task == profiles.STEADY_FLOW_PROFILE else 0,
            "cpu_model": "fixture",
            "started_at": "2026-11-25T00:00:00Z",
            "finished_at": "2026-11-25T00:01:00Z",
            "failure": None,
        }
        if kwargs is not None:
            seal["case_h5_sha256"] = common.serialization.file_sha256(write_case_h5(member_dir / "case.h5", **kwargs))
        if member["role"] == "gate":
            seal["gate"] = {"inputs_equal": True, "output_max_abs_diff": {"p": 0.0}, "output_divergences": []}
        seal.update(spec.get("seal", {}))
        common.serialization.atomic_write_json(member_dir / "seal.json", seal)
        # An outcome entry is a classification, or a (classification, fields) pair overriding the seal's accepted draws.
        for attempt, entry in enumerate(spec.get("outcomes", [seal["disposition"]] * int(seal["attempts"])), start=1):
            classification, overrides = (entry, {}) if isinstance(entry, str) else entry
            draws = {name: seal[name] for name in ("spatial_support_attempt", "schedule_acceptance_attempt")}
            outcome = record.new_outcome(member_id, attempt, classification, recorded_by="fixture", **{**draws, **overrides})
            common.serialization.atomic_write_json(record.attempt_directory(test_dir, member_id, attempt) / "outcome.json", outcome)
    return test_dir
