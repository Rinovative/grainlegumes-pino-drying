"""Protect the sampling intervention hook, the roster and record contracts, the gate comparisons, and the launcher without a solver.

Responsibilities:
  - Check an accepted override, an infeasible value, an unknown parameter, and a conditional support edge
  - Check that the scope check refuses undeclared changes and that a merge confirmation member validates
  - Check the event-weight reference, the roster version chain with its mirrored replacements, and their seal backing
  - Check that field-test paths refuse symbolic links, that cancel marks every valid node job before reporting refused
    entries, the gate's input tolerance per kind, the gate's structural output divergences, and that only a typed
    infeasible intervention is infeasible
  - Check the field-test mode of ./scripts/generation

Design principles:
  - Values come from the batch's own registry and feasible ranges, never from production constants
  - The launcher test replaces only the host boundary, so source admission and the subprocess stay out of the test

This module does NOT:
  - Generate bundles, run COMSOL, or build rosters from a Source
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import h5py
import numpy as np
import pytest
from support import field_test_fixtures
from support.generation_fixtures import lentil_generation_config

from src import common
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.cli import cli_generation_controller as controller_cli
from src.generation.field_tests import generation_field_tests_dispatch as dispatch
from src.generation.field_tests import generation_field_tests_planning as planning
from src.generation.field_tests import generation_field_tests_record as record
from src.generation.field_tests import generation_field_tests_roster as roster_service
from src.generation.field_tests import generation_field_tests_runner as runner
from src.generation.runtime import generation_runtime_node as node_service

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_STRENGTH = "permeability.anisotropy.strength"
_CHAINS = (
    ("T_in_base", "T_in_amp"),
    ("T_amb", "T_in_base"),
    ("initial_moisture.mean_db", "initial_moisture.amplitude_db"),
    ("schedule.event_duration_rel", "schedule.event_width_rel"),
)
_FAILED_GATE = "lentil-case_0001-gate"
_FRAME_A: dict[str, Any] = {
    "test_id": "t",
    "task": "steady_flow",
    "stage": "A",
    "base_source": {"campaign_name": "c", "plan_sha256": "2" * 64},
    "max_attempts": 3,
}


def _registry(config: Any) -> dict[str, Any]:
    return config.scientific_values["material"]["parameter_registry"]


def test_an_accepted_override_changes_only_the_declared_primitive_and_its_dependents(
    generation_config_factory: Callable[..., tuple[Path, Path]],
) -> None:
    config = lentil_generation_config(generation_config_factory, "steady_flow", natural_count=2)
    base = sampling.sample_case(config, 1)
    lower, upper = sampling.feasible_range(config, 1, _STRENGTH)
    value = lower + 0.25 * (upper - lower)
    assert value != base.values[_STRENGTH]
    intervened = sampling.sample_case(config, 1, intervention={_STRENGTH: value})
    assert intervened.values[_STRENGTH] == value
    sampling.require_intervention_scope(_registry(config), base.values, intervened.values, declared=frozenset({_STRENGTH}))
    assert {name for name in base.values if base.values[name] != intervened.values[name]} == {_STRENGTH}
    with pytest.raises(sampling.InterventionScopeError):
        sampling.require_intervention_scope(_registry(config), base.values, intervened.values, declared=frozenset())


@pytest.mark.parametrize("name", ["no.such.parameter", "T_init"])
def test_an_unknown_or_derived_parameter_is_refused(generation_config_factory: Callable[..., tuple[Path, Path]], name: str) -> None:
    config = lentil_generation_config(generation_config_factory, "transient_drying", natural_count=2)
    with pytest.raises(sampling.UnknownInterventionParameterError):
        sampling.sample_case(config, 1, intervention={name: 1.0})


def test_a_value_outside_the_registry_support_is_infeasible(generation_config_factory: Callable[..., tuple[Path, Path]]) -> None:
    config = lentil_generation_config(generation_config_factory, "steady_flow", natural_count=2)
    entry = _registry(config)[_STRENGTH]
    above = float(entry["upper"]) + 0.5 * (float(entry["upper"]) - float(entry["lower"]))
    with pytest.raises(sampling.InfeasibleInterventionError):
        sampling.sample_case(config, 1, intervention={_STRENGTH: above})


def test_the_feasible_range_stops_at_a_conditional_edge_and_a_value_beyond_it_is_infeasible(
    generation_config_factory: Callable[..., tuple[Path, Path]],
) -> None:
    config = lentil_generation_config(generation_config_factory, "transient_drying", natural_count=2)
    registry = _registry(config)
    # Move a chain companion to each of its feasible edges until the dependent's range is cut short of its registry support.
    edges = []
    for driver, target in _CHAINS:
        for driver_value in sampling.feasible_range(config, 1, driver):
            lower, upper = sampling.feasible_range(config, 1, target, background={driver: driver_value})
            reg_lower, reg_upper = float(registry[target]["lower"]), float(registry[target]["upper"])
            if upper < reg_upper * (1 - 1e-6):
                edges.append((driver, driver_value, target, upper, upper + 0.01 * (reg_upper - upper)))
            if lower > reg_lower * (1 + 1e-6):
                edges.append((driver, driver_value, target, lower, lower - 0.01 * (lower - reg_lower)))
    assert edges
    for driver, driver_value, target, edge, beyond in edges:
        accepted = sampling.sample_case(config, 1, intervention={driver: driver_value, target: edge})
        assert accepted.values[target] == edge
        with pytest.raises(sampling.InfeasibleInterventionError):
            sampling.sample_case(config, 1, intervention={driver: driver_value, target: beyond})


def _member(member_id: str, role: str, *, case_id: str = "case_0001", **fields: Any) -> dict[str, Any]:
    base = {
        "material_family": "lentil",
        "design_kind": "train",
        "design_index": 0,
        "case_id": case_id,
        "case_path": f"materials/lentil/{case_id}/result/case.h5",
        "case_sha256": "0" * 64,
        "case_json_sha256": "1" * 64,
    }
    defaults: dict[str, Any] = {"gate_member_id": "g", "reference_member_id": None, "row": None, "level": "nominal", "intervention": {}}
    return {"member_id": member_id, "role": role, "base": base, **defaults, **fields}


def test_a_merge_confirmation_validates_and_the_merge_level_belongs_to_a_merge_row() -> None:
    merge_row = "merge:permeability.anisotropy.max_ratio+permeability.anisotropy.strength"
    members = [
        _member("g", "gate", gate_member_id=None),
        _member("r0", "reference", reference_kind="R0", intervention={_STRENGTH: 0.5}),
        _member("m1", "challenge", reference_member_id="r0", row=merge_row, level="merge", intervention={_STRENGTH: 0.4}),
    ]
    frame: dict[str, Any] = {
        **_FRAME_A,
        "stage": "B",
        "derived_from": {"test_id": "a", "roster_sha256": "3" * 64, "verdicts_sha256": "4" * 64, "tolerances_sha256": "5" * 64},
    }
    roster = roster_service.build_roster(members=members, **frame)
    assert roster_service.validate_roster(copy.deepcopy(roster))["roster_sha256"] == roster["roster_sha256"]
    mismatched = copy.deepcopy(members)
    mismatched[2]["row"] = _STRENGTH
    with pytest.raises(ValueError, match="merge"):
        roster_service.build_roster(members=mismatched, **frame)


def test_event_shape_rows_pair_with_the_high_event_weight_reference(generation_config_factory: Callable[..., tuple[Path, Path]]) -> None:
    config = lentil_generation_config(generation_config_factory, "transient_drying", natural_count=2)
    entry = _registry(config)["schedule.component_weights"]
    references = planning._references("transient_drying", config, 1)
    weights = references["R_event"]["schedule.component_weights"]
    assert weights["event"] == pytest.approx(float(entry["maximum_each"]))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert {name for name, value in references["R_event"].items() if value != references["R0"][name]} == {"schedule.component_weights"}
    assert {"R_jitter", "R_perturb"} <= set(references)
    for row in ("schedule.event_duration_rel", "schedule.event_width_rel"):
        assert planning.reference_kinds(row) == ("R_event",)


def _chain(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish roster.json with one base and roster.v2.json replacing it by a base that mirrors its planned members."""
    first = roster_service.build_roster(members=[_member(_FAILED_GATE, "gate", gate_member_id=None)], **_FRAME_A)
    replacement = {
        "material_family": "lentil",
        "failed_case_id": "case_0001",
        "failed_gate_member_id": _FAILED_GATE,
        "failed_disposition": "solver_failure",
        "replacement_case_id": "case_0002",
    }
    mirrored = _member("lentil-case_0002-gate", "gate", case_id="case_0002", gate_member_id=None)
    second = roster_service.next_version(first, [mirrored], [replacement], version=2)
    roster_service.write_roster(directory, first)
    roster_service.write_roster(directory, second)
    return first, second


def test_a_roster_version_must_extend_its_predecessor_by_mirrored_replacement_bases(tmp_path: Path) -> None:
    directory = tmp_path / "field-test"
    first, second = _chain(directory)
    assert roster_service.load_roster_version(directory) == (second, 2)
    with pytest.raises(ValueError, match="supersede"):
        roster_service.write_roster(directory, second)

    def refused(payload: dict[str, Any], match: str, *, name: str = "roster.v2.json") -> None:
        for path in directory.glob("roster.v*.json"):
            path.unlink()
        common.serialization.atomic_write_json(directory / name, payload)
        with pytest.raises(ValueError, match=match):
            roster_service.load_roster(directory)

    tampered = copy.deepcopy(second)
    tampered["members"][0]["base"]["design_index"] = 7
    tampered["roster_sha256"] = roster_service.roster_digest(tampered)
    refused(tampered, "does not extend")
    relabelled = copy.deepcopy(second)
    relabelled["replacements"][0]["replacement_case_id"] = "case_0003"
    refused(relabelled, "digest does not match")
    refused(second, "consecutive chain", name="roster.v3.json")
    reference = _member(
        "lentil-case_0002-ref-r0",
        "reference",
        case_id="case_0002",
        gate_member_id="lentil-case_0002-gate",
        reference_kind="R0",
        intervention={_STRENGTH: 0.5},
    )
    refused(roster_service.next_version(first, [second["members"][1], reference], second["replacements"], version=2), "does not mirror")
    stray = _member("lentil-case_0003-gate", "gate", case_id="case_0003", gate_member_id=None)
    refused(roster_service.next_version(first, [second["members"][1], stray], second["replacements"], version=2), "outside its replacement bases")
    for path in directory.glob("roster.v*.json"):
        path.unlink()
    common.serialization.atomic_write_json(tmp_path / "linked.json", second)
    (directory / "roster.v2.json").symlink_to(tmp_path / "linked.json")
    with pytest.raises(ValueError, match="unsafe"):
        roster_service.load_roster(directory)


def test_only_the_sealed_failure_of_the_replaced_gate_lets_a_replacement_drop_its_base(tmp_path: Path) -> None:
    directory = tmp_path / "field-test"
    _first, second = _chain(directory)
    seal_path = record.member_directory(directory, _FAILED_GATE) / "seal.json"
    with pytest.raises(ValueError, match="not backed"):
        record.effective_members(directory, second)
    for disposition in ("succeeded", "solver_failure"):
        seal_path.unlink(missing_ok=True)
        seal = {"schema_kind": record.SEAL_SCHEMA_KIND, "member_id": _FAILED_GATE, "disposition": disposition, "gate": {"inputs_equal": True}}
        common.serialization.atomic_write_json(seal_path, seal)
        if disposition == "succeeded":
            with pytest.raises(ValueError, match="not backed"):
                record.effective_members(directory, second)
    assert [item["member_id"] for item in record.effective_members(directory, second)] == ["lentil-case_0002-gate"]


def test_field_test_paths_refuse_symbolic_links(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    (storage / "01_generation").mkdir(parents=True)
    (storage / common.paths.FIELD_TESTS_AREA_NAME).symlink_to(storage / "01_generation", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        common.paths.resolve_field_test_dir("steady_flow", "screen", storage_root=storage)
    directory = tmp_path / "field-test"
    directory.mkdir()
    (tmp_path / "outside").mkdir()
    (directory / "members").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        record.member_directory(directory, _FAILED_GATE)


def test_cancel_marks_every_valid_node_job_and_then_reports_each_refused_entry(tmp_path: Path) -> None:
    storage = tmp_path / "storage"
    directory = common.paths.resolve_field_test_dir("steady_flow", "t", storage_root=storage)
    roster_service.write_roster(directory, roster_service.build_roster(members=[_member(_FAILED_GATE, "gate", gate_member_id=None)], **_FRAME_A))
    for node in (directory / "nodes" / "a-real", directory / "nodes" / "z-real", directory / "nodes" / " padded", tmp_path / "outside"):
        (node / node_service.TICKETS_DIRECTORY).mkdir(parents=True)
    (directory / "nodes" / "linked").symlink_to(tmp_path / "outside", target_is_directory=True)
    # A node whose marker path is taken by a directory cannot be marked; the node jobs after it still are.
    (directory / "nodes" / "b-broken" / node_service.TICKETS_DIRECTORY).mkdir(parents=True)
    (directory / "nodes" / "b-broken" / node_service.CONTROL_DIRECTORY / node_service.CANCEL_MARKER).mkdir(parents=True)
    with pytest.raises(ValueError, match=r"Cancelled node jobs \['a-real', 'z-real'\]; refused node entries: .*' padded'.*b-broken.*symbolic link"):
        dispatch.cancel("steady_flow", "t", storage_root=storage)
    assert not (tmp_path / "outside" / node_service.CONTROL_DIRECTORY).exists()
    for node in ("a-real", "z-real"):
        assert (directory / "nodes" / node / node_service.CONTROL_DIRECTORY / node_service.CANCEL_MARKER).is_file()


def test_the_input_tolerance_is_relative_per_map_scalar_and_schedule_column() -> None:
    published = {"fields/Kxx": np.full((4, 4), 6.0e-9), "scalars": np.array([0.0, 293.15]), "schedule": np.array([[2.0, 1.0e5], [1.0, 2.0e5]])}
    names = sorted(published)
    noise = {name: values * (1.0 + 1.0e-14) for name, values in published.items()}
    assert runner._compare_inputs(noise, published, names)[1] == []
    drifts = {
        "fields/Kxx": published["fields/Kxx"] + 9.0e-13,  # a relative permeability change of 1.5e-4
        "scalars": np.array([1.0e-300, 293.15]),  # a sampled zero stays exactly zero
        "schedule": published["schedule"] + np.array([1.0e-11, 0.0]),  # last digits of the large column, not of the small one
    }
    for name, drifted in drifts.items():
        assert runner._compare_inputs({**published, name: drifted}, published, names)[1] == [name]


def test_the_gate_output_comparison_flags_every_structural_divergence(tmp_path: Path) -> None:
    kwargs = field_test_fixtures.case_kwargs("transient_drying", 0)
    published = field_test_fixtures.write_case_h5(tmp_path / "published.h5", **kwargs)
    variants: dict[str, tuple[dict[str, Any], list[str]]] = {
        "same": ({}, []),
        "censored": ({"wet_final": 0.2}, ["reached"]),
        "shifted": ({"times": kwargs["times"] + 0.5}, ["hour_axis"]),
        "short": ({"times": kwargs["times"][:3], "states": kwargs["states"][:3]}, ["shape:global/values", "shape:time", "shape:transient/fields"]),
    }
    for name, (change, expected) in variants.items():
        produced = field_test_fixtures.write_case_h5(tmp_path / f"{name}.h5", **{**kwargs, **change})
        assert runner._compare_outputs(produced, published)[1] == expected, name
    # As published, the exact stop after the last stored hour adds its own datasets and a diagnostics row. The solver does not
    # reproduce it bitwise; a drift that keeps that layout is measured, not structural.
    stop = float(kwargs["times"][-1]) + 0.4
    for name, t_stop in (("base_stop", stop), ("drifted_stop", stop + 6.5e-5)):
        path = field_test_fixtures.write_case_h5(tmp_path / f"{name}.h5", **{**kwargs, "t_final": t_stop})
        with h5py.File(path, "r+") as handle:
            handle.create_dataset("exact_stop/time", data=np.array([t_stop]))
            handle.create_dataset("exact_stop/fields", data=kwargs["states"][-1].astype(np.float32))
    differences, divergences = runner._compare_outputs(tmp_path / "drifted_stop.h5", tmp_path / "base_stop.h5")
    assert divergences == []
    assert differences["exact_stop/time"] == pytest.approx(6.5e-5, rel=1.0e-6)
    stripped = field_test_fixtures.write_case_h5(tmp_path / "stripped.h5", **kwargs)
    with h5py.File(stripped, "r+") as handle:
        del handle["scalar/values"]
        states = handle["transient/fields"]
        assert isinstance(states, h5py.Dataset)
        states[0, 0, 0, 0] = np.nan
    assert runner._compare_outputs(stripped, published)[1] == ["missing:scalar/values", "nonfinite:transient/fields"]


def test_only_a_typed_infeasible_intervention_is_infeasible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        message = "The generator refused the bundle."
        raise ValueError(message)

    monkeypatch.setattr(runner.case_service, "generate_case_input_bundle", refuse)
    with pytest.raises(runner._Stop) as stopped:
        runner._generate_bundle(cast("Any", None), 1, {"role": "challenge", "intervention": {_STRENGTH: 0.5}}, tmp_path / "bundle")
    assert (stopped.value.classification, stopped.value.reason_code) == ("binding_refused", "bundle_generation")


class _Host:
    """Record the one host call a field-test command makes instead of running a subprocess."""

    calls: ClassVar[list[tuple[tuple[str, ...], dict[str, Any]]]] = []

    def __init__(self, _layout: Any, admission: Any) -> None:
        self.admission = admission

    def call(self, *arguments: str, **options: Any) -> str:
        _Host.calls.append((arguments, options))
        return "{}\n"


@pytest.mark.parametrize(("action", "options", "frozen"), [("submit", [], True), ("submit", ["--dry-run"], False)])
def test_the_generation_entry_point_runs_field_tests_and_freezes_only_for_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str, options: list[str], frozen: bool
) -> None:
    storage = tmp_path / "storage"
    freezes: list[bool] = []

    def admit(_layout: Any, *, freeze: bool) -> Any:
        freezes.append(freeze)
        return SimpleNamespace(frozen=freeze, uncommitted=False, commit="a" * 40, fingerprint="b" * 64, tree_id="t")

    monkeypatch.setattr(controller_cli.host_service.HostLayout, "resolve", classmethod(lambda _cls, _repository: SimpleNamespace(storage=storage)))
    monkeypatch.setattr(controller_cli.host_service, "admit_source", admit)
    monkeypatch.setattr(controller_cli.host_service, "GenerationHost", _Host)
    monkeypatch.setattr(_Host, "calls", [])
    test_dir = common.paths.resolve_field_test_dir("steady_flow", "screen", storage_root=storage)
    assert controller_cli.main(["field-test", action, str(test_dir), *options]) == 0
    assert freezes == [frozen]
    [(arguments, keywords)] = _Host.calls
    assert arguments == (action, "steady_flow", "screen", *options, "--storage-root", str(storage))
    assert keywords == {"scheduled": False, "module": "src.generation.cli.cli_field_test"}
    assert controller_cli.main(["field-test", action, str(tmp_path / "elsewhere" / "steady_flow" / "screen")]) == 2
