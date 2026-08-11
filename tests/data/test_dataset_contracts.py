# ruff: noqa: S101
"""Authoritative steady and transient Dataset contract inspection."""

from dataclasses import FrozenInstanceError

import pytest

from src import domain, generation
from src.datasets.runtime import dataset_runtime_transient as transient_runtime
from src.datasets.contracts import dataset_contracts_transient as transient_contract
from src.datasets.contracts import dataset_contracts_views as views

_STEADY_DIGEST = "d40dc74f5f8e70dc19a7e592e4d720ff27ca6131b70bd64d88556791566fac0a"
_TRANSIENT_DIGEST = "73ff7473076e5ed7fc4f64bd13bffba53a8bdb34c52f0e9bf351e603f72206b2"


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
    assert transient_contract.transient_contract_payload()["dt"] == {"value": 1.0, "unit": "h"}
    assert not hasattr(transient_runtime, "transient_contract_payload")
