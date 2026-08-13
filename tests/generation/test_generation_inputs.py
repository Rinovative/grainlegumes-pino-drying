# ruff: noqa: S101, PLR2004, SLF001
"""Final configuration, naming, ownership, sampling, and domain contracts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import yaml

from src import common, datasets, domain, generation
from src.datasets.packages import dataset_packages_planning as package_planning
from src.generation.cases import generation_cases_fields as fields
from src.generation.cases import generation_cases_sampling as sampling
from src.generation.cases import generation_cases_schedule as schedule_service
from src.generation.cases import generation_cases_seeding as seeding
from src.generation.contracts import generation_contracts_materials as materials
from src.generation.contracts import generation_contracts_provenance as provenance_service
from src.generation.publication import generation_publication_inventory as inventory

if TYPE_CHECKING:
    from collections.abc import Callable


def test_semantic_seed_derivation_is_version_bound() -> None:
    """Protect persisted random-substream identity across internal ownership changes."""
    assert (
        seeding.derive_seed(
            12345,
            "generation_batch",
            "profile_a",
            "material_a",
            "regime_a",
            "evaluation_a",
        )
        == 2107745129
    )


def test_generation_provenance_is_compact_controlled_and_source_resolved() -> None:
    """Validate the canonical scientific provenance vocabulary and references."""
    root = Path.cwd()
    registry = yaml.safe_load((root / "configs/generation/registry.yaml").read_text(encoding="utf-8"))
    source_config = yaml.safe_load((root / "configs/generation/sources.yaml").read_text(encoding="utf-8"))
    source_keys = {source["source_key"] for source in source_config["sources"]}
    assert registry["schema_version"] == 1
    assert set(registry) == {"schema_kind", "schema_version", "parameters"}

    allowed = {"evidence", "source_refs", "method", "verification", "applicability", "note"}
    provenance_count = 0

    def visit(value: Any) -> None:
        nonlocal provenance_count
        if isinstance(value, dict):
            if {"evidence", "source_refs"}.issubset(value):
                provenance_count += 1
                assert set(value).issubset(allowed)
                assert value["evidence"] in provenance_service.EVIDENCE_CLASSES
                assert set(value["source_refs"]).issubset(source_keys)
                if "verification" in value:
                    assert value["verification"] == provenance_service.REPRODUCED_VERIFICATION
                if "applicability" in value:
                    assert value["applicability"]
                    assert set(value["applicability"]).issubset(provenance_service.APPLICABILITY_KEYS)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for config_path in sorted((root / "configs/generation").rglob("*.yaml")):
        visit(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    assert provenance_count > 0
    with pytest.raises(ValueError, match="source_refs must identify evidence"):
        provenance_service.validate_provenance(
            {"evidence": "literature_direct", "source_refs": []},
            sources={},
            label="source-free literature",
        )


def test_generation_public_facade_is_explicit_and_narrow() -> None:
    """Protect the lazy same-level package and orchestration surface."""
    expected = {
        "benchmark",
        "campaign",
        "cases",
        "contracts",
        "publication",
        "readiness",
        "runtime",
        "smoke",
        "validation",
        "workflow",
    }
    assert set(generation.__all__) == expected
    assert all(getattr(generation, name) is not None for name in expected)
    assert "cli" not in generation.__all__
    assert not hasattr(generation, "config")
    assert not hasattr(generation, "storage")


def test_current_authoritative_configs_resolve_reviewed_outputs() -> None:
    """Record current config-derived outputs without making them validator rules."""
    expected = {
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"): (1200, 28),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"): (660, 54),
        Path("configs/generation/campaigns/steady_flow/technical_smoke.yaml"): (2, 28),
        Path("configs/generation/campaigns/transient_drying/technical_smoke.yaml"): (2, 54),
        Path("configs/generation/campaigns/transient_drying/pilot_check.yaml"): (18, 54),
    }
    for path, (case_count, dimension) in expected.items():
        campaign = generation.cases.config.load_campaign_config(path, require_executable=False)
        assert campaign.total_case_count == case_count
        assert sum(campaign.batches[0].scientific_values["sampling"]["block_dimensions"].values()) == dimension
        inventory_names = tuple(family for role in generation.contracts.material_roles() for family in campaign.material_roles[role])
        assert len(inventory_names) == len(set(inventory_names))
        assert set(inventory_names) <= set(generation.contracts.available_material_families())
    pilot_path = Path("configs/generation/campaigns/transient_drying/pilot_check.yaml")
    fast = generation.cases.config.load_campaign_config(
        pilot_path,
        require_executable=False,
        pilot_cases_per_material=1,
    )
    assert fast.total_case_count == 6

    for path in expected:
        campaign = generation.cases.config.load_campaign_config(path)
        exports = campaign.batches[0].scientific_values["output_contract"]["exports"]
        assert all(export.get("pattern") and set(export["columns"]) == set(export["units"]) for export in exports)
        assert all({"state", "verified"}.isdisjoint(export) for export in exports)


def test_profile_schema_rejects_obsolete_verification_field(
    generation_config_factory: Any,
) -> None:
    """Reject the old persistent verification model instead of migrating it."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    profile_path = config_path.parent / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["exports"][0]["source"] = {
        "state": "obsolete",
        "pattern": profile["exports"][0]["source"],
    }
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match="obsolete persistent export-mapping verification state",
    ):
        generation.cases.config.load_campaign_config(config_path)


def test_incomplete_mapping_declaration_is_discovery_only(
    generation_config_factory: Any,
) -> None:
    """Permit structural discovery only when executable mappings are not required."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    profile_path = config_path.parent / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    profile["exports"][0]["columns"]["x"] = None
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")

    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    assert "x" not in campaign.batches[0].scientific_values["output_contract"]["exports"][0]["columns"]
    with pytest.raises(
        generation.cases.config.GenerationConfigError,
        match="incomplete; an explicit executable mapping declaration is required",
    ):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize("option", ["--exclusive", "--reservation=reserved", "--nodelist=node-a"])
def test_execution_rejects_nonordinary_scheduler_options(
    generation_config_factory: Any,
    option: str,
) -> None:
    """Prevent execution YAML from reintroducing exclusivity or reservations."""
    config_path, _template = generation_config_factory(scheduler_kind="slurm")
    execution_path = config_path.parent / "execution.yaml"
    execution = yaml.safe_load(execution_path.read_text(encoding="utf-8"))
    execution["cluster"]["scheduler_options"] = [option]
    execution_path.write_text(yaml.safe_dump(execution, sort_keys=False), encoding="utf-8")

    with pytest.raises(generation.cases.config.GenerationConfigError, match="allocation directives"):
        generation.cases.config.load_campaign_config(config_path)


def test_parameter_ood_allocation_covers_profile_eligible_units_evenly() -> None:
    """Protect exact one-unit OOD assignments derived from profile applicability."""
    paths = (
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
    )
    for path in paths:
        campaign = generation.cases.config.load_campaign_config(path, require_executable=False)
        for batch in (item for item in campaign.batches if item.sampling_regime == "parameter_ood"):
            policy = batch.scientific_values["parameter_ood"]
            eligible = policy["eligible_units"]
            allocation = policy["case_allocation"]
            counts = policy["allocation_counts"]
            assert len(allocation) == len(batch.case_indices)
            assert set(counts) == {unit["unit_id"] for unit in eligible}
            assert min(counts.values()) >= 1
            assert max(counts.values()) - min(counts.values()) <= 1
            registry = batch.scientific_values["material"]["parameter_registry"]
            assert {unit["ood_group"] for unit in eligible} == set(materials.active_ood_groups(registry, campaign.profile.id))
            for case_index in (batch.case_indices[0], batch.case_indices[-1]):
                assignment = batch.case_assignment(case_index)
                sample = sampling.sample_case(batch, case_index)
                assert sample.ood_provenance["active_unit_id"] == assignment["ood_unit_id"]
                assert sample.ood_provenance["active_ood_group"] == assignment["ood_group"]
                assert sample.ood_provenance["units_per_case"] == 1


def test_generation_and_hdf5_versions_are_integer_one(
    generation_config_factory: Any,
) -> None:
    """Reject pre-dataset named version strings at the owning config loader."""
    config_path, _template = generation_config_factory()
    batch = generation.cases.config.load_campaign_config(config_path).batches[0]
    versions = (
        batch.scientific_values["generator_version"],
        batch.scientific_values["storage"]["schema_version"],
        batch.scientific_values["storage"]["converter_version"],
    )
    assert versions == (1, 1, 1)
    assert all(type(version) is int for version in versions)

    common_path = config_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common["generator_version"] = "named-version"
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="generator_version must be an integer"):
        generation.cases.config.load_campaign_config(config_path)

    common["generator_version"] = 1
    common["storage"]["converter_version"] = "named-version"
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="converter_version must be an integer"):
        generation.cases.config.load_campaign_config(config_path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("rtol", 0.0, "rtol must be positive"),
        ("rtol", -1.0e-4, "rtol must be positive"),
        ("atol", -1.0e-10, "atol must be non-negative"),
        ("rtol", float("nan"), "must be a finite real value"),
        ("atol", float("inf"), "must be a finite real value"),
        ("rtol", "1e-4", "must be a finite real value"),
        ("atol", True, "must be a finite real value"),
        ("rtol", 1.0e-3, "no greater than"),
        ("atol", 1.0e-6, "no greater than"),
    ],
)
def test_transient_initial_state_tolerance_validation(
    generation_config_factory: Any,
    key: str,
    value: object,
    message: str,
) -> None:
    """Reject malformed or unsafe semantic initial-state tolerances."""
    config_path, _template = generation_config_factory()
    common_path = config_path.parent / "common.yaml"
    common = yaml.safe_load(common_path.read_text(encoding="utf-8"))
    common["validation"]["transient_initial_state"][key] = value
    common_path.write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")

    with pytest.raises(generation.cases.config.GenerationConfigError, match=message):
        generation.cases.config.load_campaign_config(config_path)


def test_transient_validation_tolerance_has_scoped_identity_ownership(
    generation_config_factory: Any,
) -> None:
    """Bind admission provenance without changing inputs or steady batches."""
    transient_path, _template = generation_config_factory()
    original_transient = generation.cases.config.load_campaign_config(transient_path).batches[0]
    transient_common_path = transient_path.parent / "common.yaml"
    transient_common = yaml.safe_load(transient_common_path.read_text(encoding="utf-8"))
    transient_common["validation"]["transient_initial_state"]["rtol"] = 5.0e-5
    transient_common_path.write_text(yaml.safe_dump(transient_common, sort_keys=False), encoding="utf-8")
    changed_transient = generation.cases.config.load_campaign_config(transient_path).batches[0]

    assert changed_transient.scientific_config_digest != original_transient.scientific_config_digest
    assert changed_transient.batch_id != original_transient.batch_id
    assert changed_transient.case_input_config_digest == original_transient.case_input_config_digest

    steady_path, _template = generation_config_factory(simulation_profile="steady_flow")
    original_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]
    steady_common_path = steady_path.parent / "common.yaml"
    steady_common = yaml.safe_load(steady_common_path.read_text(encoding="utf-8"))
    steady_common["validation"]["transient_initial_state"]["rtol"] = 5.0e-5
    steady_common_path.write_text(yaml.safe_dump(steady_common, sort_keys=False), encoding="utf-8")
    changed_steady = generation.cases.config.load_campaign_config(steady_path).batches[0]

    assert "validation" not in original_steady.scientific_values
    assert changed_steady.scientific_config_digest == original_steady.scientific_config_digest
    assert changed_steady.batch_id == original_steady.batch_id
    assert changed_steady.case_input_config_digest == original_steady.case_input_config_digest


def _temporary_family_campaign(
    tmp_path: Path,
    name: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    """Copy one layered campaign while retaining repository scientific owners."""
    directory = tmp_path / name
    directory.mkdir()
    source = Path("configs/generation/campaigns/transient_drying/family_generalization.yaml")
    campaign = yaml.safe_load(source.read_text(encoding="utf-8"))
    reference_keys = (
        "sources_config",
        "registry_config",
        "operations_config",
        "profile_config",
    )
    for key in reference_keys:
        campaign[key] = str(Path(campaign[key]).resolve())

    common_path = directory / "common.yaml"
    common = yaml.safe_load(Path(campaign["common_config"]).read_text(encoding="utf-8"))
    campaign["common_config"] = str(common_path)

    execution_path = directory / "execution.yaml"
    execution = yaml.safe_load(Path(campaign["execution_config"]).read_text(encoding="utf-8"))
    campaign["execution_config"] = str(execution_path)

    campaign_path = directory / "campaign.yaml"
    return campaign_path, campaign, common_path, common, execution_path, execution


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write one deterministic temporary YAML fixture."""
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def test_valid_config_edits_are_resolved_without_source_synchronization(
    tmp_path: Path,
) -> None:
    """Prove counts, seeds, roles, packages, science, and resources are config-owned."""
    campaign_path, campaign, common_path, common, execution_path, execution = _temporary_family_campaign(tmp_path, "editable")
    campaign["sampling"]["seed_base"] = 12345
    campaign["membership"] = {
        "seed": 12346,
        "per_seen_material": {
            "train": 190,
            "validation": 25,
            "id_test": 25,
        },
    }
    for material_family in campaign["sampling"]["counts"]["natural"]:
        campaign["sampling"]["counts"]["natural"][material_family] = 241 if material_family in campaign["material_roles"]["seen"] else 81
    for material_family in campaign["sampling"]["counts"]["parameter_ood"]:
        campaign["sampling"]["counts"]["parameter_ood"][material_family] = 81

    original_seen = campaign["material_roles"]["seen"][1]
    replacement_seen = campaign["material_roles"]["near_family_ood"][0]
    campaign["material_roles"]["seen"][1] = replacement_seen
    campaign["material_roles"]["near_family_ood"][0] = original_seen
    parameter_counts = campaign["sampling"]["counts"]["parameter_ood"]
    parameter_counts[replacement_seen] = parameter_counts.pop(original_seen)
    natural_counts = campaign["sampling"]["counts"]["natural"]
    natural_counts[replacement_seen] = 241
    natural_counts[original_seen] = 81
    campaign["dataset_packages"] = [package for package in campaign["dataset_packages"] if package["evaluation_regime"] != "extreme_family_ood"]

    common["grid"].update({"nx": 17, "ny": 11, "Lx": 1.6, "Ly": 1.0, "Lz": 0.4})
    common["time"].update({"start": 0.0, "stop": 84.0, "interval": 0.5})
    common["scientific_fixed_values"]["T_flow_ref"]["value"] = 301.15
    common["scientific_fixed_values"]["p_ref"]["value"] = 100000.0
    common["storage"].update(
        {
            "compression_level": 6,
            "chunk_time": 3,
            "chunk_y": 5,
            "chunk_x": 7,
        }
    )

    execution["runtime"].update(
        {
            "timeout_seconds": 4200,
            "maximum_failures": 3,
        }
    )
    execution["submission"].update(
        {
            "pending_buffer": 2,
            "poll_interval_seconds": 7,
            "max_running_cases": 3,
        }
    )
    execution["cluster"]["cores_per_case"] = 8
    execution["site"].update(
        {
            "cpu_host": "synthetic-host",
            "cores_per_node": 16,
            "python_module": "Python/test",
            "comsol_module": "Comsol/test",
        }
    )
    _write_yaml(common_path, common)
    _write_yaml(execution_path, execution)
    _write_yaml(campaign_path, campaign)

    resolved = generation.cases.config.load_campaign_config(
        campaign_path,
        require_executable=False,
    )
    expected_total = sum(count for regime_counts in campaign["sampling"]["counts"].values() for count in regime_counts.values())
    assert resolved.total_case_count == expected_total
    assert resolved.membership == campaign["membership"]
    assert resolved.batches[0].scientific_values["campaign_seed"] == 12345
    assert resolved.material_roles["seen"][1] == replacement_seen
    expected_evaluation_regimes = tuple(dict.fromkeys(str(package["evaluation_regime"]) for package in campaign["dataset_packages"]))
    assert resolved.evaluation_regimes == expected_evaluation_regimes
    assert (
        resolved.require_batch(
            material_family=replacement_seen,
            sampling_regime="parameter_ood",
        ).material_family
        == replacement_seen
    )
    assert (
        resolved.find_batch(
            material_family=original_seen,
            sampling_regime="parameter_ood",
        )
        is None
    )
    scientific = resolved.batches[0].scientific_values
    assert scientific["grid"]["dx"] == pytest.approx(0.1)
    assert scientific["grid"]["dy"] == pytest.approx(0.1)
    assert scientific["time"]["regular_times"] == [index * 0.5 for index in range(169)]
    assert scientific["scientific_fixed_values"]["T_flow_ref"] == 301.15
    assert scientific["scientific_fixed_values"]["p_ref"] == 100000.0
    assert scientific["storage"]["compression_level"] == 6
    assert resolved.execution_values["runtime"]["timeout_seconds"] == 4200.0
    assert resolved.execution_values["runtime"]["maximum_failures"] == 3
    assert resolved.execution_values["cluster"]["cores_per_case"] == 8
    assert resolved.execution_values["submission"] == {
        "pending_buffer": 2,
        "poll_interval_seconds": 7,
        "max_running_cases": 3,
    }
    assert resolved.execution_values["site"]["cpu_host"] == "synthetic-host"


def test_sampling_coordinate_order_is_config_owned_and_identity_bound(
    tmp_path: Path,
) -> None:
    """Bind explicit registry coordinate order through each scientific batch identity."""
    campaign_path, campaign, common_path, common, execution_path, execution = _temporary_family_campaign(
        tmp_path,
        "coordinate-order",
    )
    registry_path = campaign_path.parent / "registry.yaml"
    registry = yaml.safe_load(Path(campaign["registry_config"]).read_text(encoding="utf-8"))
    campaign["registry_config"] = str(registry_path)
    _write_yaml(common_path, common)
    _write_yaml(execution_path, execution)
    _write_yaml(registry_path, registry)
    _write_yaml(campaign_path, campaign)

    original = generation.cases.config.load_campaign_config(campaign_path, require_executable=False)
    original_family = original.material_inventory[0]
    original_batch = original.require_batch(
        material_family=original_family,
        sampling_regime="natural",
    )
    original_parameters = original_batch.scientific_values["sampling"]["blocks"]["airflow"]["parameters"]

    left = "bed.structure.cross_scale_corr"
    right = "bed.structure.fine_ani_x"
    left_order = registry["parameters"][left]["sampling_order"]
    right_order = registry["parameters"][right]["sampling_order"]
    registry["parameters"][left]["sampling_order"] = right_order
    registry["parameters"][right]["sampling_order"] = left_order
    _write_yaml(registry_path, registry)

    reordered = generation.cases.config.load_campaign_config(campaign_path, require_executable=False)
    reordered_batch = reordered.require_batch(
        material_family=original_family,
        sampling_regime="natural",
    )
    reordered_parameters = reordered_batch.scientific_values["sampling"]["blocks"]["airflow"]["parameters"]

    assert original_parameters.index(left) == reordered_parameters.index(right)
    assert original_parameters.index(right) == reordered_parameters.index(left)
    assert reordered_batch.scientific_config_digest != original_batch.scientific_config_digest
    assert reordered_batch.batch_id != original_batch.batch_id
    assert reordered.campaign_digest != original.campaign_digest


def test_invalid_config_combinations_report_authoritative_owner(
    tmp_path: Path,
) -> None:
    """Reject invalid counts, membership, role overlap, and package sources precisely."""

    def expect_failure(
        name: str,
        mutate: Callable[[dict[str, Any]], None],
        expected_key: str | Callable[[dict[str, Any]], str],
    ) -> None:
        campaign_path, campaign, common_path, common, execution_path, execution = _temporary_family_campaign(tmp_path, name)
        mutate(campaign)
        _write_yaml(common_path, common)
        _write_yaml(execution_path, execution)
        _write_yaml(campaign_path, campaign)
        with pytest.raises(generation.cases.config.GenerationConfigError) as caught:
            generation.cases.config.load_campaign_config(
                campaign_path,
                require_executable=False,
            )
        details = generation.cases.config.validation_error_details(campaign_path, caught.value)
        assert details["file"] == str(campaign_path)
        assert details["owner_to_edit"] == str(campaign_path)
        resolved_expected_key = expected_key(campaign) if callable(expected_key) else expected_key
        assert details["key"] == resolved_expected_key
        assert details["actual_value"] != "<missing>"
        assert details["expected_type_or_rule"]

    expect_failure(
        "bad-count",
        lambda campaign: campaign["sampling"]["counts"]["natural"].__setitem__(
            next(iter(campaign["sampling"]["counts"]["natural"])),
            0,
        ),
        lambda campaign: f"sampling.counts.natural.{next(iter(campaign['sampling']['counts']['natural']))}",
    )
    expect_failure(
        "membership-overflow",
        lambda campaign: campaign["membership"]["per_seen_material"].__setitem__("train", 999),
        "membership.per_seen_material",
    )
    expect_failure(
        "role-overlap",
        lambda campaign: campaign["material_roles"]["near_family_ood"].append(campaign["material_roles"]["seen"][0]),
        "material_roles",
    )

    def mismatch_near_family_source(campaign: dict[str, Any]) -> None:
        declaration = next(item for item in campaign["dataset_packages"] if item["evaluation_regime"] == "near_family_ood")
        declaration["source_role"] = "far_family_ood"

    def near_family_package_key(campaign: dict[str, Any]) -> str:
        index = next(index for index, item in enumerate(campaign["dataset_packages"]) if item["evaluation_regime"] == "near_family_ood")
        return f"dataset_packages[{index}]"

    expect_failure(
        "missing-package-source",
        mismatch_near_family_source,
        near_family_package_key,
    )


def test_readiness_distinguishes_failed_static_sentinels_from_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report an executed failing scientific sentinel as a launch blocker."""
    monkeypatch.setattr(
        generation.readiness.sentinel_service,
        "run_static_sentinels",
        lambda *_args: {"status": "blocked_by_scientific_sanity_guard"},
    )
    report = generation.readiness.build_readiness_report(
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
        run_static_sentinels=True,
    )
    assert "STATIC_GENERATOR_SENTINELS_BLOCKED" in report["status_lines"]
    assert report["production_ready_for_user_launch"] is False


def test_resolved_parameter_inspection_exposes_atomic_provenance_and_coordinates() -> None:
    """Expose true design coordinates and inherited complete-record provenance."""
    transient = generation.cases.config.load_campaign_config(
        Path("configs/generation/campaigns/transient_drying/family_generalization.yaml"),
        require_executable=False,
    )
    report = next(iter(inventory.audit_campaign(transient).values()))
    assert len(report.sampled_coordinate_names) == report.total_effective_dimension
    assert report.total_effective_dimension > 0
    assert "schedule.component_weights[1]" in report.sampled_coordinate_names
    assert "schedule.component_weights[2]" in report.sampled_coordinate_names

    oswin = inventory.inspect_campaign_parameter(transient, "oswin")
    oswin_component = inventory.inspect_campaign_parameter(transient, "A_osw")
    density = inventory.inspect_campaign_parameter(transient, "density_calibration")
    density_component = inventory.inspect_campaign_parameter(transient, "eps_bed_cal_ref")
    kinetics = inventory.inspect_campaign_parameter(transient, "two_compartment_kinetics")
    simplex = inventory.inspect_campaign_parameter(transient, "schedule_simplex")
    schedule_bound = inventory.inspect_campaign_parameter(transient, "T_in_min")
    clip_bound = inventory.inspect_campaign_parameter(transient, "phi_clip_min")
    permeability_mean = inventory.inspect_campaign_parameter(transient, "kappa_mean")
    template_fixed = inventory.inspect_campaign_parameter(transient, "cp_w")
    grid_spacing = inventory.inspect_campaign_parameter(transient, "grid.dx")
    wall_coefficient = inventory.inspect_campaign_parameter(transient, "U_wall")
    time_stop = inventory.inspect_campaign_parameter(transient, "time.stop")
    density_formula = inventory.inspect_campaign_parameter(
        transient,
        "physical_formulas.rho_bu_dry",
    )
    porosity_support = inventory.inspect_campaign_parameter(
        transient,
        "packing_porosity_mean_support",
    )
    assert oswin_component["atomic_record"] == "oswin"
    assert density_component["atomic_record"] == "density_calibration"
    assert density["coordinate_labels"] == ["rho_bu_dry_ref"]
    assert kinetics["coordinate_labels"] == ["r_surf_0", "r_int_surf", "f_surf"]
    assert simplex["coordinate_labels"] == [
        "schedule.component_weights[1]",
        "schedule.component_weights[2]",
    ]
    assert "generation.cases.generation_cases_schedule.generate_schedule" in simplex["producer_to_consumer_path"]["effective_downstream_consumers"]
    assert schedule_bound["producer_to_consumer_path"]["runtime_mapping_state"] == "generator_consumed"
    assert clip_bound["producer_to_consumer_path"]["runtime_mapping_state"] == ("generator_consumed_and_template_fixed_requires_native_verification")
    assert clip_bound["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation.cases.generation_cases_schedule feasibility or psychrometric conversion",
        "canonical COMSOL template fixed physics; Python has no runtime setter",
    ]
    assert permeability_mean["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "generation.cases.generation_cases_fields._permeability_fields",
        "generation.cases.generation_cases_fields._porosity_field",
    ]
    assert template_fixed["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_fixed_no_python_runtime_setter")
    assert set(template_fixed["provenance"]).issubset({"evidence", "source_refs", "method", "verification", "applicability", "note", "sources"})
    authored_common = yaml.safe_load(Path("configs/generation/common.yaml").read_text(encoding="utf-8"))
    expected_dx = authored_common["grid"]["Lx"] / (authored_common["grid"]["nx"] - 1)
    assert grid_spacing["configured"] == {"dx": expected_dx}
    assert grid_spacing["provenance"]["evidence"] == "derived"
    assert grid_spacing["provenance"]["verification"] == "mathematically_reproduced"
    fixed = authored_common["scientific_fixed_values"]
    expected_u_wall = 1.0 / (1.0 / fixed["h_ext"]["value"] + fixed["d_wall"]["value"] / fixed["k_wall"]["value"])
    assert wall_coefficient["configured"]["value"] == pytest.approx(expected_u_wall)
    assert wall_coefficient["provenance"]["verification"] == "mathematically_reproduced"
    assert "dx" not in authored_common["grid"]
    assert "dy" not in authored_common["grid"]
    assert "value" not in authored_common["scientific_fixed_values"]["U_wall"]
    assert time_stop["configured"] == {"stop": authored_common["time"]["stop"]}
    assert density_formula["configured"] == {"rho_bu_dry": "rho_bu_dry_ref*(1-eps_bed)/(1-eps_bed_cal_ref)"}
    assert density_formula["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_formula_requires_native_model_verification")
    assert set(porosity_support["materials"]) == set(transient.material_inventory)
    assert all(not material["applicability"]["parameter_ood"] for material in porosity_support["materials"].values())
    for family in transient.material_inventory:
        assert oswin_component["materials"][family]["provenance"] == oswin["materials"][family]["provenance"]
        assert density_component["materials"][family]["provenance"] == density["materials"][family]["provenance"]

    steady = generation.cases.config.load_campaign_config(
        Path("configs/generation/campaigns/steady_flow/family_generalization.yaml"),
        require_executable=False,
    )
    steady_reference_pressure = inventory.inspect_campaign_parameter(steady, "p_ref")
    assert steady_reference_pressure["producer_to_consumer_path"]["effective_downstream_consumers"] == [
        "canonical steady COMSOL template fixed conditioning"
    ]
    assert steady_reference_pressure["producer_to_consumer_path"]["runtime_mapping_state"] == ("template_fixed_no_python_runtime_setter")
    with pytest.raises(ValueError, match="not applicable to profile 'steady_flow'"):
        inventory.inspect_campaign_parameter(steady, "density_calibration")
    with pytest.raises(ValueError, match="not applicable to profile 'steady_flow'"):
        inventory.inspect_campaign_parameter(steady, "time.stop")


def test_scalar_handoff_rejects_an_unknown_name(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Protect the exact scalar schema after a modified file receives a fresh hash."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    config = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    bundle = generation.cases.case.generate_case_input_bundle(config, 1, tmp_path / "case")
    scalar_path = bundle.directory / "scalars.csv"
    content = scalar_path.read_text(encoding="utf-8")
    assert bundle.scalar_handoff is not None
    assert bundle.scalar_handoff.field_names == generation.contracts.profiles.TRANSIENT_SCALAR_INPUT_FIELDS
    assert len(bundle.scalar_handoff.entries) == 12
    non_handoff_fields = {"T_init", "T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}
    assert non_handoff_fields.isdisjoint(bundle.scalar_handoff.field_names)
    assert all(f"{name};" not in content for name in non_handoff_fields)
    assert bundle.case_payload["sampled_values"]["T_init"] == bundle.case_payload["sampled_values"]["T_amb"]
    assert {"T_flow_ref", "p_ref", "p_out", "f_wet_dm_max"}.isdisjoint(bundle.case_payload["sampled_values"])
    registry_entry = config.scientific_values["material"]["parameter_registry"]["T_init"]
    assert registry_entry["kind"] == "derived"
    assert registry_entry["derivation"] == "copy"
    assert registry_entry["sources"] == ["T_amb"]
    field_prefix = "T_amb;"
    assert content.count(field_prefix) == 1
    scalar_path.write_text(content.replace(field_prefix, "unexpected_flow_temperature;"), encoding="utf-8")
    payload = copy.deepcopy(bundle.case_payload)
    payload["input_files"]["scalars.csv"] = {
        "sha256": common.serialization.file_sha256(scalar_path),
        "size_bytes": scalar_path.stat().st_size,
    }
    with pytest.raises(ValueError, match="missing, duplicate, unknown"):
        generation.contracts.scalar_handoff.admit_case_scalar_handoff(
            payload,
            bundle.directory,
        )


def test_each_sampled_morphology_control_changes_its_owned_field(generation_config_factory: Any) -> None:
    """Protect material consumption of all 17 independent morphology controls."""
    config_path, _template = generation_config_factory()
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    sample = sampling.sample_case(batch, 1)
    values = sample.values
    registry = batch.scientific_values["material"]["parameter_registry"]
    grid = {"Lx": 1.2, "Ly": 0.75, "Lz": 0.8, "dx": 0.015, "dy": 0.015, "nx": 81, "ny": 51}
    seeds = {"bed": 101, "pressure_bc": 202, "initial_moisture": 303}
    family_contract = batch.scientific_values["material"]
    family_bounds = family_contract["initial_moisture_bounds"]
    porosity_support = family_contract["packing_porosity_mean_support"]
    baseline = fields.generate_spatial_fields(
        "transient_drying",
        grid,
        values,
        seeds=seeds,
        family_bounds=family_bounds,
        packing_porosity_mean_support=porosity_support,
        material_kappa_nominal=float(registry["kappa_mean"]["nominal"]),
        active_ood_unit=None,
    )
    bed_names = (
        "bed.structure.coarse_len_rel",
        "bed.structure.fine_len_rel",
        "bed.structure.coarse_weight",
        "bed.structure.cross_scale_corr",
        "bed.structure.fine_ani_x",
        "bed.structure.fine_ani_y",
        "bed.perturbations.amplitude",
        "bed.perturbations.granularity",
        "bed.perturbations.sign_bias",
    )
    moisture_names = (
        "initial_moisture.mean_db",
        "initial_moisture.amplitude_db",
        "initial_moisture.structure.coarse_len_rel",
        "initial_moisture.structure.fine_len_rel",
        "initial_moisture.structure.coarse_weight",
        "initial_moisture.structure.cross_scale_corr",
        "initial_moisture.structure.fine_ani_x",
        "initial_moisture.structure.fine_ani_y",
    )
    observed: dict[str, float] = {}
    for name in (*bed_names, *moisture_names):
        variant = copy.deepcopy(values)
        if name == "initial_moisture.mean_db":
            amplitude = float(values["initial_moisture.amplitude_db"])
            candidates = (float(family_bounds["lower"]) + amplitude, float(family_bounds["upper"]) - amplitude)
        elif name == "initial_moisture.amplitude_db":
            mean = float(values["initial_moisture.mean_db"])
            feasible_upper = min(
                float(registry[name]["upper"]),
                mean - float(family_bounds["lower"]),
                float(family_bounds["upper"]) - mean,
            )
            candidates = (float(registry[name]["lower"]), feasible_upper)
        else:
            candidates = (float(registry[name]["lower"]), float(registry[name]["upper"]))
        variant[name] = max(candidates, key=lambda value: abs(value - float(values[name])))
        if name == "bed.structure.coarse_weight":
            variant["bed.structure.fine_weight"] = 1.0 - variant[name]
        elif name == "initial_moisture.structure.coarse_weight":
            variant["initial_moisture.structure.fine_weight"] = 1.0 - variant[name]
        generated = fields.generate_spatial_fields(
            "transient_drying",
            grid,
            variant,
            seeds=seeds,
            family_bounds=family_bounds,
            packing_porosity_mean_support=porosity_support,
            material_kappa_nominal=float(registry["kappa_mean"]["nominal"]),
            active_ood_unit=None,
        )
        affected_fields = ("Kxx", "Kxy", "Kyy", "eps_bed") if name in bed_names else ("X_0_db_field",)
        difference = max(float(np.max(np.abs(generated.columns[field] - baseline.columns[field]))) for field in affected_fields)
        scale = max(float(np.max(np.abs(baseline.columns[field]))) for field in affected_fields)
        observed[name] = difference / max(scale, np.finfo(np.float64).tiny)
    assert set(observed) == set(bed_names) | set(moisture_names)
    assert all(effect > 1e-10 for effect in observed.values()), observed


def test_domain_moisture_and_dataset_membership() -> None:
    """Protect one moisture implementation and deterministic ID membership."""
    dry_basis = np.asarray([0.0, 0.25, 1.0])
    wet_basis = domain.moisture.dry_basis_to_wet_basis(dry_basis)
    np.testing.assert_allclose(wet_basis, [0.0, 0.2, 0.5])
    np.testing.assert_allclose(domain.moisture.wet_basis_to_dry_basis(wet_basis), dry_basis)
    water = domain.moisture.granular_water_content([1.0, 2.0], [3.0, 4.0], 0.5)
    np.testing.assert_allclose(water, [2.0, 3.0])
    np.testing.assert_allclose(domain.moisture.dry_basis_moisture(water, [20.0, 30.0]), [0.1, 0.1])
    assert domain.moisture.bulk_wet_basis_moisture(water, [20.0, 30.0]) == pytest.approx(5.0 / 55.0)

    candidates = [
        {
            "material_family": "lentil",
            "package_case_id": f"lentil/case_{index}",
            "case_input_id": character * 64,
        }
        for index, character in enumerate("abcdef", start=1)
    ]
    plan = {
        "evaluation_regime": "id",
        "materials": ["lentil"],
        "membership_seed": 19,
        "membership_counts_per_material": {"train": 2, "validation": 1, "id_test": 1},
    }
    membership_a = package_planning._shared_id_membership(copy.deepcopy(plan), copy.deepcopy(candidates))
    membership_b = package_planning._shared_id_membership(
        copy.deepcopy(plan),
        list(reversed(copy.deepcopy(candidates))),
    )
    assert membership_a == membership_b
    assert len(membership_a) == 4
    assert list(membership_a.values()).count("train") == 2
    assert list(membership_a.values()).count("validation") == 1
    assert list(membership_a.values()).count("id_test") == 1


def test_steady_conditioning_audit_rejects_hidden_case_varying_solver_input(
    generation_config_factory: Any,
) -> None:
    """Require every varying stationary dependency to be a declared task input."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow")
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    contract = copy.deepcopy(batch.scientific_values["steady_flow_conditioning"])
    record = {
        "simulation_profile": "steady_flow",
        "steady_flow_conditioning": contract,
    }

    accepted = package_planning.audit_steady_flow_conditioning([record])
    assert accepted["hidden_conditioning"] is False
    assert accepted["T_flow_ref_owner"] == "package_fixed"
    coupled_record = copy.deepcopy(record)
    coupled_record["simulation_profile"] = "transient_drying"
    mixed = package_planning.audit_steady_flow_conditioning([record, coupled_record])
    assert mixed["source_profiles"] == ["steady_flow", "transient_drying"]
    assert mixed["conditioning_contract_digest"] == accepted["conditioning_contract_digest"]

    hidden = copy.deepcopy(record)
    dependency = next(item for item in hidden["steady_flow_conditioning"]["dependencies"] if item["name"] == "T_flow_ref")
    dependency["affects_stationary_solution"] = True
    dependency["owner"] = "model_input"
    with pytest.raises(ValueError, match=r"Hidden steady-flow conditioning.*T_flow_ref"):
        package_planning.audit_steady_flow_conditioning([hidden])


def test_parameter_ood_eligibility_uses_registry_dependency_blocks() -> None:
    """Include airflow OOD in both views while excluding transient-only changes from steady."""
    config_path = Path("configs/generation/campaigns/transient_drying/family_generalization.yaml")
    campaign = generation.cases.config.load_campaign_config(config_path, require_executable=False)
    batch = campaign.require_batch(
        material_family="lentil",
        sampling_regime="parameter_ood",
    )
    registry = batch.scientific_values["material"]["parameter_registry"]

    def candidate(parameter: str) -> dict[str, Any]:
        entry = registry[parameter]
        return {
            "package_case_id": f"case__{parameter}",
            "batch": batch,
            "case_payload": {
                "ood": {
                    "group": entry["ood_group"],
                    "selected_units": [parameter],
                    "units_per_case": 1,
                    "selections": {
                        parameter: {
                            "selection_kind": "scalar_interval",
                            "transformed_gap": 0.2,
                        }
                    },
                },
                "sampled_values": {parameter: 1.0},
                "coupled_selections": {},
                "block_provenance": {entry["block"]: {"design": "synthetic"}},
            },
        }

    airflow = candidate("pressure_bc.mean")
    steady_eligible, steady_parameters, steady_evidence, _reason = package_planning._ood_eligibility(
        airflow,
        view=datasets.contracts.views.get_view("steady_flow"),
    )
    transient_eligible, transient_parameters, _transient_evidence, _reason = package_planning._ood_eligibility(
        airflow,
        view=datasets.contracts.views.get_view("transient_drying"),
    )
    assert steady_eligible is transient_eligible is True
    assert steady_parameters == transient_parameters == ("pressure_bc.mean",)
    assert steady_evidence["parameters"][0]["block"] == "airflow"

    moisture = candidate("initial_moisture.mean_db")
    eligible, parameters, evidence, reason = package_planning._ood_eligibility(
        moisture,
        view=datasets.contracts.views.get_view("steady_flow"),
    )
    assert eligible is False
    assert parameters == ()
    assert evidence["group"] == "initial_moisture"
    assert "steady_flow" in str(reason)
    assert (
        package_planning._ood_eligibility(
            moisture,
            view=datasets.contracts.views.get_view("transient_drying"),
        )[0]
        is True
    )


def test_duplicate_source_policy_and_id_ood_overlap_are_explicit() -> None:
    """Resolve matched physical inputs deterministically and reject train/OOD leakage."""
    physical_id = "c" * 64
    candidates = [
        {
            "package_case_id": "steady_case",
            "simulation_case_id": "1" * 64,
            "case_input_id": physical_id,
            "simulation_profile": "steady_flow",
            "material_family": "lentil",
        },
        {
            "package_case_id": "transient_case",
            "simulation_case_id": "2" * 64,
            "case_input_id": physical_id,
            "simulation_profile": "transient_drying",
            "material_family": "lentil",
        },
    ]
    with pytest.raises(ValueError, match="requires an explicit source preference"):
        package_planning.resolve_duplicate_case_inputs(
            candidates,
            dataset_view="steady_flow",
            policy="reject_duplicates",
        )

    selected, decisions = package_planning.resolve_duplicate_case_inputs(
        list(reversed(candidates)),
        dataset_view="steady_flow",
        policy="prefer_transient_source",
    )
    assert [candidate["package_case_id"] for candidate in selected] == ["transient_case"]
    assert decisions[0]["selected_simulation_case_id"] == "2" * 64
    assert decisions[0]["excluded_simulation_case_ids"] == ["1" * 64]

    id_package = package_planning.PreparedPackage(
        plan={"evaluation_regime": "id"},
        batch_records=[],
        candidates=[
            {
                "simulation_case_id": "3" * 64,
                "case_input_id": physical_id,
                "dataset_membership": "train",
            }
        ],
        excluded=[],
        membership={},
        source_decisions=[],
        steady_conditioning=None,
    )
    ood_package = package_planning.PreparedPackage(
        plan={"evaluation_regime": "parameter_ood"},
        batch_records=[],
        candidates=[
            {
                "simulation_case_id": "4" * 64,
                "case_input_id": physical_id,
                "dataset_membership": "parameter_ood",
            }
        ],
        excluded=[],
        membership={},
        source_decisions=[],
        steady_conditioning=None,
    )
    with pytest.raises(ValueError, match="ID training and OOD package source overlap"):
        package_planning._validate_no_id_ood_overlap((id_package, ood_package))


def test_heater_schedule_retries_complete_realization_deterministically(
    generation_config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an infeasible whole schedule and replay the accepted retry exactly."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying")
    batch = generation.cases.config.load_generation_config(
        config_path,
        only_batch=generation.cases.config.build_batch_name(
            "transient_drying",
            "lentil",
            "natural",
        ),
    )
    sample = sampling.sample_case(batch, 1)
    assert batch.seed_base is not None
    seeds = {name: seeding.derive_seed(batch.seed_base, "case", "1", name) for name in ("schedule_shared", "schedule_independent")}
    original = schedule_service._feasibility_reason
    call_count = 0

    def force_first_rejection(*args: Any, **kwargs: Any) -> str | None:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 1:
            return "forced deterministic whole-schedule rejection"
        return original(*args, **kwargs)

    monkeypatch.setattr(schedule_service, "_feasibility_reason", force_first_rejection)
    first = schedule_service.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        batch.scientific_values["scientific_fixed_values"],
        seeds=seeds,
    )
    second = schedule_service.generate_schedule(
        sample.values,
        batch.scientific_values["time"],
        batch.scientific_values["scientific_fixed_values"],
        seeds=seeds,
    )
    np.testing.assert_array_equal(first.values, second.values)
    assert first.metadata == second.metadata
    assert first.metadata["schedule_rejection_count"] == 1
    assert first.metadata["schedule_acceptance_attempt"] == 2
    assert first.metadata["min_heater_temperature_rise"] >= 0.0
    assert 0.0 < first.metadata["min_phi_source_air"] <= first.metadata["max_phi_source_air"] <= 1.0
    assert first.metadata["column_order"] == [field.name for field in generation.contracts.get_profile_contract("transient_drying").schedule_fields]
    assert first.metadata["phi_source_air_usage"] == "validation_and_provenance_only"


def test_schedule_feasibility_diagnostic_uses_resolved_operating_bounds() -> None:
    """Report the effective humidity envelope without freezing production values."""
    fixed = {
        "T_in_min": 290.0,
        "T_in_max": 310.0,
        "omega_min": 0.001,
        "omega_max": 0.02,
        "phi_operational_min": 0.2,
        "phi_operational_max": 0.7,
    }
    reason = schedule_service._feasibility_reason(
        np.asarray([300.0]),
        np.asarray([0.01]),
        np.asarray([0.8]),
        np.asarray([0.5]),
        ambient_temperature=295.0,
        fixed=fixed,
    )

    assert reason == "phi_in_bc violates the configured operating envelope [0.2, 0.7]"


def test_all_generation_source_references_resolve() -> None:
    """Require every authored source reference to resolve through the source registry."""
    registry = yaml.safe_load(Path("configs/generation/sources.yaml").read_text(encoding="utf-8"))
    sources = registry["sources"]
    source_keys = {record["source_key"] for record in sources}
    assert len(source_keys) == len(sources)
    used_refs: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"source_ref", "source_refs"}:
                    refs = [item] if isinstance(item, str) else item
                    if isinstance(refs, list):
                        used_refs.update(ref for ref in refs if isinstance(ref, str))
                else:
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    for path in Path("configs/generation").rglob("*.yaml"):
        if path.name != "sources.yaml":
            collect(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert used_refs <= source_keys
