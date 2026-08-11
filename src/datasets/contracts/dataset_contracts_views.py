"""
===============================================================================
dataset_contracts_views.py
===============================================================================
Define buildable Dataset views and uniform immutable contract inspection.
Responsibilities:
  - Register steady-flow and transient-drying Dataset-view identities
  - Normalize authoritative learned and transition contracts for inspection
  - Expose ordered fields, tensor groups, and runtime-resolved shape semantics
Design principles:
  - Trainable-task registration remains owned by ``src.domain.tasks``
  - Generation supplies source-profile, membership, and regime vocabulary
  - Inspection reuses authoritative digests and never serializes a third contract
This module does NOT:
  - Build packages, load HDF5 data, or register transient training behavior
  - Infer parameter relevance from filenames or mutable campaign membership
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias, cast

from src import domain, generation

from . import dataset_contracts_identity as identity
from . import dataset_contracts_transient as transient_contract

DatasetViewId = Literal["steady_flow", "transient_drying"]
PackageRegime: TypeAlias = generation.contracts.EvaluationRegime
IdMembership: TypeAlias = generation.contracts.IdMembership
DatasetFieldPurpose = Literal["conditioning", "target", "archive"]
SpatialShapeSemantics = Literal["runtime_dataset_identity", "runtime_source_hdf5"]
TemporalSemantics = Literal["static_snapshot", "fixed_step_transition"]
TargetSemantics = Literal["direct_task_outputs", "next_state_minus_current_state"]
DatasetContract = domain.tasks.spec.TaskSpec | transient_contract.TransientStepContract

PACKAGE_REGIMES: Final = cast("tuple[PackageRegime, ...]", generation.contracts.evaluation_regimes())
ID_MEMBERSHIPS: Final = cast("tuple[IdMembership, ...]", generation.contracts.id_memberships())
TECHNICAL_SMOKE_MEMBERSHIP: Final = "technical_smoke"

_PROFILE_CONTRACTS: Final = tuple(
    generation.contracts.get_profile_contract(profile_id) for profile_id in generation.contracts.available_profile_ids()
)
_SPATIAL_SAMPLE_LAYOUT: Final = ("channel", "y", "x")
_CHANNEL_SAMPLE_LAYOUT: Final = ("channel",)


@dataclass(frozen=True, slots=True)
class DatasetViewSpec:
    """Describe one buildable Dataset view and its source relevance contract."""

    id: DatasetViewId
    registered_task_id: str | None
    source_profiles: tuple[str, ...]
    parameter_ood_blocks: tuple[str, ...]
    contract_digest: str

    @property
    def trainable(self) -> bool:
        """Return whether the view is backed by a registered learning task."""
        return self.registered_task_id is not None


@dataclass(frozen=True, slots=True)
class DatasetFieldInspection:
    """Describe one ordered field through the uniform Dataset inspection API."""

    name: str
    unit: str
    role: str
    representation: str | None
    source_name: str | None


@dataclass(frozen=True, slots=True)
class DatasetGroupInspection:
    """Describe one ordered tensor group at the model-facing sample boundary."""

    name: str
    purpose: DatasetFieldPurpose
    fields: tuple[DatasetFieldInspection, ...]
    tensor_layout: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetTemporalInspection:
    """Describe authoritative temporal coordinates and supported sample modes."""

    fields: tuple[DatasetFieldInspection, ...]
    tensor_dtype: str
    regular_transition_step: float
    time_unit: str
    authoritative_source: str
    configured_horizon_source: str
    exact_stop_usage: str
    boundary_interval_interpolation: str
    boundary_interval_representation: str
    sampling_modes: tuple[transient_contract.TransientSampleMode, ...]


@dataclass(frozen=True, slots=True)
class DatasetContractInspection:
    """Expose one authoritative contract through uniform immutable semantics."""

    view: DatasetViewSpec
    contract: DatasetContract
    groups: tuple[DatasetGroupInspection, ...]
    tensor_dtype: str
    temporal_semantics: TemporalSemantics
    temporal: DatasetTemporalInspection | None
    storage_representation: str
    target_semantics: TargetSemantics
    target_derivation: str
    spatial_shape_semantics: SpatialShapeSemantics

    @property
    def contract_digest(self) -> str:
        """Return the exact persisted compatibility digest for this view."""
        return self.view.contract_digest

    @property
    def fields(self) -> tuple[DatasetFieldInspection, ...]:
        """Return every inspected state, conditioning, target, and archive field."""
        return tuple(field for group in self.groups for field in group.fields)

    @property
    def time_step(self) -> float | None:
        """Return the regular transition duration when this view is temporal."""
        return None if self.temporal is None else self.temporal.regular_transition_step

    @property
    def time_unit(self) -> str | None:
        """Return the physical time unit when this view is temporal."""
        return None if self.temporal is None else self.temporal.time_unit

    @property
    def sampling_modes(self) -> tuple[transient_contract.TransientSampleMode, ...]:
        """Return explicit runtime sampling modes supported by this view."""
        return () if self.temporal is None else self.temporal.sampling_modes

    def group(self, name: str) -> DatasetGroupInspection:
        """Return one exact inspected tensor group by name."""
        for group in self.groups:
            if group.name == name:
                return group
        available = ", ".join(group.name for group in self.groups)
        message = f"Dataset view {self.view.id!r} has no group {name!r}. Available groups: {available}."
        raise ValueError(message)


def _source_profiles(view_id: DatasetViewId) -> tuple[str, ...]:
    """Return profiles that advertise one learning view in canonical order."""
    return tuple(contract.id for contract in _PROFILE_CONTRACTS if view_id in contract.available_learning_views)


def _steady_contract_digest() -> str:
    """Return the registered task's learned-data contract digest."""
    return domain.tasks.registry.get_task("steady_flow").data_contract_digest


def _steady_field(field: domain.tasks.spec.FieldSpec) -> DatasetFieldInspection:
    """Project one authoritative TaskSpec field without changing its semantics."""
    return DatasetFieldInspection(
        name=field.name,
        unit=field.unit,
        role=field.role,
        representation=field.representation,
        source_name=field.source_name,
    )


def _transient_fields(
    fields: tuple[transient_contract.DataField, ...],
    *,
    role: str,
) -> tuple[DatasetFieldInspection, ...]:
    """Project one authoritative transient field group without copying policy."""
    return tuple(
        DatasetFieldInspection(
            name=field.name,
            unit=field.unit,
            role=role,
            representation=None,
            source_name=None,
        )
        for field in fields
    )


def _steady_inspection(view: DatasetViewSpec, task: domain.tasks.spec.TaskSpec) -> DatasetContractInspection:
    """Normalize the registered steady TaskSpec for generic inspection."""
    sample_layout = tuple(axis for axis in task.tensor_layout if axis != "batch")
    output_fields = {field.name: field for field in task.outputs}
    groups = [
        DatasetGroupInspection(
            name="inputs",
            purpose="conditioning",
            fields=tuple(_steady_field(field) for field in task.inputs),
            tensor_layout=sample_layout,
        )
    ]
    groups.extend(
        DatasetGroupInspection(
            name=group.id,
            purpose="target",
            fields=tuple(_steady_field(output_fields[name]) for name in group.fields),
            tensor_layout=sample_layout,
        )
        for group in task.output_groups
    )
    return DatasetContractInspection(
        view=view,
        contract=task,
        groups=tuple(groups),
        tensor_dtype=identity.TRAINING_TENSOR_DTYPE,
        temporal_semantics="static_snapshot",
        temporal=None,
        storage_representation="task_declared_field_representations",
        target_semantics="direct_task_outputs",
        target_derivation="generated_reference_fields",
        spatial_shape_semantics="runtime_dataset_identity",
    )


def _transient_inspection(
    view: DatasetViewSpec,
    contract: transient_contract.TransientStepContract,
) -> DatasetContractInspection:
    """Normalize the transient step contract for generic inspection."""
    groups = (
        DatasetGroupInspection(
            "state",
            "conditioning",
            _transient_fields(contract.dynamic_state, role="dynamic_state"),
            _SPATIAL_SAMPLE_LAYOUT,
        ),
        DatasetGroupInspection(
            "static",
            "conditioning",
            _transient_fields(contract.static_spatial_conditioning, role="static_spatial_conditioning"),
            _SPATIAL_SAMPLE_LAYOUT,
        ),
        DatasetGroupInspection(
            "boundary",
            "conditioning",
            _transient_fields(contract.step_boundary_conditioning, role="step_boundary_conditioning"),
            _CHANNEL_SAMPLE_LAYOUT,
        ),
        DatasetGroupInspection(
            "scalars",
            "conditioning",
            _transient_fields(contract.scalar_conditioning, role="scalar_conditioning"),
            _CHANNEL_SAMPLE_LAYOUT,
        ),
        DatasetGroupInspection(
            "target",
            "target",
            _transient_fields(contract.target_increments, role="target_increment"),
            _SPATIAL_SAMPLE_LAYOUT,
        ),
        DatasetGroupInspection(
            "archived_ablation",
            "archive",
            _transient_fields(contract.archived_ablation_fields, role="archived_ablation"),
            _SPATIAL_SAMPLE_LAYOUT,
        ),
    )
    return DatasetContractInspection(
        view=view,
        contract=contract,
        groups=groups,
        tensor_dtype=contract.tensor_dtype,
        temporal_semantics="fixed_step_transition",
        temporal=DatasetTemporalInspection(
            fields=_transient_fields(contract.temporal.fields, role="temporal_coordinate"),
            tensor_dtype=contract.temporal.tensor_dtype,
            regular_transition_step=contract.time_step,
            time_unit=contract.time_unit,
            authoritative_source=contract.temporal.authoritative_source,
            configured_horizon_source=contract.temporal.configured_horizon_source,
            exact_stop_usage=contract.temporal.exact_stop_usage,
            boundary_interval_interpolation=contract.boundary_interval_interpolation,
            boundary_interval_representation=contract.boundary_interval_representation,
            sampling_modes=contract.sampling_modes,
        ),
        storage_representation=contract.canonical_storage_representation,
        target_semantics="next_state_minus_current_state",
        target_derivation=contract.target_derivation_stage,
        spatial_shape_semantics="runtime_source_hdf5",
    )


_VIEWS: Final = MappingProxyType(
    {
        "steady_flow": DatasetViewSpec(
            id="steady_flow",
            registered_task_id="steady_flow",
            source_profiles=_source_profiles("steady_flow"),
            parameter_ood_blocks=("airflow",),
            contract_digest=_steady_contract_digest(),
        ),
        "transient_drying": DatasetViewSpec(
            id="transient_drying",
            registered_task_id=None,
            source_profiles=_source_profiles("transient_drying"),
            parameter_ood_blocks=(),
            contract_digest=transient_contract.transient_contract_digest(),
        ),
    }
)


def available_views() -> tuple[DatasetViewId, ...]:
    """Return buildable Dataset-view identifiers in deterministic order."""
    return tuple(cast("DatasetViewId", name) for name in sorted(_VIEWS))


def get_view(view_id: str) -> DatasetViewSpec:
    """Resolve one exact buildable Dataset-view identifier."""
    try:
        return _VIEWS[view_id]
    except KeyError as error:
        available = ", ".join(available_views())
        message = f"Unknown dataset view {view_id!r}. Available views: {available}."
        raise ValueError(message) from error


def inspect_contract(view_id: str) -> DatasetContractInspection:
    """
    Return one uniform immutable Dataset-view contract inspection.

    Parameters
    ----------
    view_id : str
        Exact buildable Dataset-view identifier.

    Returns
    -------
    DatasetContractInspection
        Ordered field groups, model-facing tensor layouts, dtype, temporal,
        storage, target, spatial-shape, digest, and authoritative raw contract.

    Notes
    -----
    This projection is not serialized or hashed. Steady and transient digests
    remain owned by ``TaskSpec`` and ``src.datasets.contracts.transient`` respectively.

    """
    view = get_view(view_id)
    if view.registered_task_id is not None:
        return _steady_inspection(view, domain.tasks.registry.get_task(view.registered_task_id))
    return _transient_inspection(view, transient_contract.TRANSIENT_STEP_CONTRACT)


def validate_view_for_profile(view_id: str, profile_id: str) -> DatasetViewSpec:
    """Resolve a view and require the selected simulation profile to provide it."""
    view = get_view(view_id)
    if profile_id not in view.source_profiles:
        message = f"Dataset view {view.id!r} is unavailable from simulation profile {profile_id!r}."
        raise ValueError(message)
    return view
