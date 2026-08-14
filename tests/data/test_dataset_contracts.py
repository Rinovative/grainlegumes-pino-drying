# ruff: noqa: S101, SLF001
"""Authoritative steady and transient Dataset contract inspection."""

from dataclasses import FrozenInstanceError

import pytest

from src import domain, generation
from src.datasets.contracts import dataset_contracts_identity as dataset_identity
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_builder as package_builder
from src.datasets.packages import dataset_packages_trajectory as trajectory
from src.datasets.runtime import dataset_runtime_transient as transient_runtime
from src.generation.cases import generation_cases_case as case_contract

_STEADY_DIGEST = "d40dc74f5f8e70dc19a7e592e4d720ff27ca6131b70bd64d88556791566fac0a"
_TRANSIENT_DIGEST = "ee84455fb6265aba26abe19c3f0167226aad9b672d2228096cb74e59be36d077"


def test_package_payload_schema_identity_is_view_specific() -> None:
    """Keep steady package identity independent of transient-index evolution."""
    steady = package_builder._schema_identity("steady_flow")
    transient = package_builder._schema_identity("transient_drying")
    assert steady["transient_index"] == dataset_identity.TRAINING_DATASET_SCHEMA_VERSION
    assert transient["transient_index"] == trajectory.TRANSIENT_INDEX_SCHEMA_VERSION
    assert steady["generation_case"] == transient["generation_case"] == case_contract.CASE_CONTRACT_DIGEST
    assert {key: value for key, value in steady.items() if key != "transient_index"} == {
        key: value for key, value in transient.items() if key != "transient_index"
    }


def test_contract_inspection_is_uniform_immutable_and_preserves_persisted_identity() -> None:
    """Expose both authoritative contracts through one ordered immutable DTO."""
    steady = views.inspect_contract("steady_flow")
    steady_task = domain.tasks.registry.get_task("steady_flow")
    assert steady.contract is steady_task
    assert steady.contract_digest == steady_task.data_contract_digest == _STEADY_DIGEST
    assert tuple(group.name for group in steady.groups) == ("inputs", *(group.id for group in steady_task.output_groups))
    assert tuple(field.name for field in steady.group("inputs").fields) == steady_task.input_names
    assert tuple(field.name for group in steady.groups if group.purpose == "target" for field in group.fields) == steady_task.output_names
    assert all(group.tensor_layout == steady_task.tensor_layout[1:] for group in steady.groups)
    assert steady.tensor_dtype == "float32"
    assert (steady.temporal_semantics, steady.time_step, steady.time_unit) == ("static_snapshot", None, None)
    assert steady.target_semantics == "direct_task_outputs"
    assert steady.spatial_shape_semantics == "runtime_dataset_identity"

    transient = views.inspect_contract("transient_drying")
    step = transient_contract.TRANSIENT_STEP_CONTRACT
    assert type(transient) is type(steady) is views.DatasetContractInspection
    assert transient.contract is step
    assert transient.contract_digest == transient_contract.transient_contract_digest() == _TRANSIENT_DIGEST
    assert tuple(group.name for group in transient.groups) == (
        "state",
        "static",
        "boundary",
        "scalars",
        "target",
        "archived_ablation",
    )
    assert tuple(field.name for field in transient.group("state").fields) == tuple(field.name for field in step.dynamic_state)
    assert tuple(field.name for field in transient.group("target").fields) == tuple(field.name for field in step.target_increments)
    assert transient.group("boundary").tensor_layout == ("channel",)
    assert transient.group("target").tensor_layout == ("channel", "y", "x")
    assert transient.tensor_dtype == step.tensor_dtype
    assert (transient.temporal_semantics, transient.time_step, transient.time_unit) == (
        "fixed_step_transition",
        step.time_step,
        step.time_unit,
    )
    assert transient.temporal is not None
    assert tuple(field.name for field in transient.temporal.fields) == ("t_n", "t_n_plus_1", "dt")
    assert transient.temporal.authoritative_source == "canonical_hdf5_regular_time_axis"
    assert transient.temporal.configured_horizon_source == "generation_scientific_config.time.stop"
    assert transient.temporal.boundary_interval_interpolation == "linear_between_boundary_schedule_support_nodes"
    assert transient.temporal.boundary_interval_representation == "regular_endpoints_plus_optional_startup_support_without_extra_training_timestep"
    assert transient.sampling_modes == ("one_step_transition", "rollout_window")
    assert transient.storage_representation == step.canonical_storage_representation
    assert transient.target_semantics == "next_state_minus_current_state"
    assert transient.target_derivation == step.target_derivation_stage
    assert transient.spatial_shape_semantics == "runtime_source_hdf5"
    assert transient.fields == tuple(field for group in transient.groups for field in group.fields)
    assert not hasattr(transient.contract, "spatial_shape")
    with pytest.raises(FrozenInstanceError):
        transient.groups[0].tensor_layout = ("channel",)  # type: ignore[misc]


def test_transient_contract_derives_source_fields_and_owns_persisted_serializer() -> None:
    """Bind Dataset channel selection to Generation names/units without a runtime duplicate."""
    source = generation.contracts.get_profile_contract(transient_contract.TRANSIENT_PROFILE_ID)
    contract = transient_contract.TRANSIENT_STEP_CONTRACT
    assert tuple((field.name, field.unit) for field in contract.dynamic_state) == tuple((field.name, field.unit) for field in source.transient_fields)
    for field in (
        *contract.static_spatial_conditioning,
        *contract.scalar_conditioning,
        *contract.archived_ablation_fields,
    ):
        source_field = source.field(field.name)
        assert (field.name, field.unit) == (source_field.name, source_field.unit)
    learned_scalar_names = tuple(field.name for field in contract.scalar_conditioning)
    assert learned_scalar_names == (
        "r_surf_0",
        "r_int_surf",
        "f_surf",
        "A_osw",
        "B_osw",
        "C_osw",
        "k_gr",
        "cp_gr_dry",
    )
    assert tuple(name for name in generation.contracts.profiles.TRANSIENT_SCALAR_INPUT_FIELDS if name not in learned_scalar_names) == (
        "T_amb",
        "eps_bed_cal_ref",
        "rho_bu_dry_ref",
        "X_target_wb",
    )
    assert {field.name for field in contract.static_spatial_conditioning}.issuperset({"eps_bed", "rho_bu_dry"})
    assert "T_amb" in {field.name for field in contract.step_boundary_conditioning}
    assert tuple(field.name for field in contract.step_boundary_conditioning[-5:]) == (
        "startup_support_time_offset",
        "T_in_bc_startup_support",
        "omega_in_bc_startup_support",
        "phi_in_bc_startup_support",
        "startup_support_present",
    )
    assert contract.temporal.exact_stop_usage == "diagnostic_only_no_training_transition_or_rollout"
    payload = transient_contract.transient_contract_payload()
    assert payload["time"]["fields"] == [
        {"name": "t_n", "unit": "h"},
        {"name": "t_n_plus_1", "unit": "h"},
        {"name": "dt", "unit": "h"},
    ]
    assert payload["time"]["configured_horizon_source"] == "generation_scientific_config.time.stop"
    assert payload["boundary_interval"] == {
        "interpolation": "linear_between_boundary_schedule_support_nodes",
        "representation": "regular_endpoints_plus_optional_startup_support_without_extra_training_timestep",
    }
    assert not {"interval_mean", "interval_integral", "interval_sequence"}.intersection(field["name"] for field in payload["boundary"])
    assert payload["sampling"]["modes"] == ["one_step_transition", "rollout_window"]
    assert "dt" not in payload
    assert not hasattr(transient_runtime, "transient_contract_payload")
