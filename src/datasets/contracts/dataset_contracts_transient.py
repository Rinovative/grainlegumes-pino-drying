"""
dataset_contracts_transient.py

Define and serialize the unregistered transient drying sample-data contract.
Responsibilities:
  - Select Dataset-owned conditioning fields from Generation storage descriptors
  - Declare ordered state, temporal, boundary, scalar, target, and archive fields
  - Validate explicit sampling and boundary-preserving spatial-view specifications
  - Serialize and digest the exact persisted transient sample contract
Design principles:
  - Generation owns canonical HDF5 source names and physical units
  - Dataset owns model-facing channel selection and delta-target derivation
  - HDF5 regular-time coordinates, not copied index values, own physical time
This module does NOT:
  - Register a transient learning task or build transient tensors
  - Validate HDF5 payloads, normalize time, or execute model rollouts
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypeAlias, cast

from src import generation

TransientSampleMode: TypeAlias = Literal["one_step_transition", "rollout_window"]

TRANSIENT_PROFILE_ID: Final = "transient_drying"
TRANSIENT_VIEW_ID: Final = "transient_drying"
TRANSIENT_VIEW_CONTRACT_SCHEMA_VERSION: Final = 1
TRANSIENT_SPATIAL_REPRESENTATION_SCHEMA_VERSION: Final = 1
TRANSIENT_SAMPLE_MODES: Final[tuple[TransientSampleMode, ...]] = (
    "one_step_transition",
    "rollout_window",
)
_SPATIAL_AXIS_COUNT: Final = 2
_MINIMUM_SPATIAL_AXIS_LENGTH: Final = 2


def validate_spatial_stride(value: Any) -> int:
    """Return one exact positive spatial stride without coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = "spatial_stride must be an integer."
        raise TypeError(message)
    if value < 1:
        message = "spatial_stride must be >= 1."
        raise ValueError(message)
    return value


@dataclass(frozen=True, slots=True)
class TransientSpatialRepresentation:
    """Describe one exact endpoint-preserving view of a canonical grid."""

    source_shape: tuple[int, int]
    spatial_stride: int
    represented_shape: tuple[int, int]
    y_indices: tuple[int, ...]
    x_indices: tuple[int, ...]
    _index_identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject shape, stride, endpoint, or index-map drift."""
        stride = validate_spatial_stride(self.spatial_stride)
        source_shape = _validate_spatial_shape(
            self.source_shape,
            label="source_shape",
        )
        expected_y = tuple(range(0, source_shape[0], stride))
        expected_x = tuple(range(0, source_shape[1], stride))
        represented_shape = (len(expected_y), len(expected_x))
        if represented_shape != self.represented_shape:
            message = "represented_shape does not match the source grid and spatial stride."
            raise ValueError(message)
        if self.y_indices != expected_y or self.x_indices != expected_x:
            message = "Transient spatial indices must be the exact endpoint-preserving strided source indices."
            raise ValueError(message)
        if expected_y[-1] != source_shape[0] - 1 or expected_x[-1] != source_shape[1] - 1:
            message = "Transient spatial indices must retain both physical endpoints."
            raise ValueError(message)
        object.__setattr__(self, "_index_identity_sha256", self._compute_index_identity_sha256())

    @property
    def index_identity_sha256(self) -> str:
        """Return the exact source-index-map identity."""
        return self._index_identity_sha256

    def _compute_index_identity_sha256(self) -> str:
        """Compute the immutable exact source-index-map identity once."""
        encoded = json.dumps(
            {
                "schema_version": TRANSIENT_SPATIAL_REPRESENTATION_SCHEMA_VERSION,
                "source_shape": list(self.source_shape),
                "spatial_stride": self.spatial_stride,
                "represented_shape": list(self.represented_shape),
                "y_indices": list(self.y_indices),
                "x_indices": list(self.x_indices),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the exact index-based spatial representation."""
        return {
            "schema_version": TRANSIENT_SPATIAL_REPRESENTATION_SCHEMA_VERSION,
            "source_shape": list(self.source_shape),
            "spatial_stride": self.spatial_stride,
            "represented_shape": list(self.represented_shape),
            "axis_indices": {
                "y": list(self.y_indices),
                "x": list(self.x_indices),
            },
            "index_identity_sha256": self.index_identity_sha256,
            "representation_transformation": "endpoint_preserving_strided_source_index_selection",
        }


def _validate_spatial_shape(value: Any, *, label: str) -> tuple[int, int]:
    """Return one exact two-axis spatial shape."""
    if (
        not isinstance(value, tuple)
        or len(value) != _SPATIAL_AXIS_COUNT
        or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < _MINIMUM_SPATIAL_AXIS_LENGTH for axis in value)
    ):
        message = f"{label} must contain two integer spatial axes >= 2."
        raise ValueError(message)
    return cast("tuple[int, int]", value)


def resolve_spatial_representation(
    canonical_shape: tuple[int, int],
    spatial_stride: int,
) -> TransientSpatialRepresentation:
    """Return one exact endpoint-preserving canonical-grid representation."""
    stride = validate_spatial_stride(spatial_stride)
    source_shape = _validate_spatial_shape(
        canonical_shape,
        label="canonical_shape",
    )
    if any((axis - 1) % stride != 0 for axis in source_shape):
        message = f"spatial_stride={stride} does not preserve both boundaries of canonical spatial shape {source_shape}."
        raise ValueError(message)
    y_indices = tuple(range(0, source_shape[0], stride))
    x_indices = tuple(range(0, source_shape[1], stride))
    return TransientSpatialRepresentation(
        source_shape=source_shape,
        spatial_stride=stride,
        represented_shape=(len(y_indices), len(x_indices)),
        y_indices=y_indices,
        x_indices=x_indices,
    )


def resolve_spatial_view(
    canonical_shape: tuple[int, int],
    spatial_stride: int,
) -> tuple[int, int]:
    """Return the boundary-preserving effective ``(Y, X)`` shape."""
    return resolve_spatial_representation(
        canonical_shape,
        spatial_stride,
    ).represented_shape


@dataclass(frozen=True, slots=True)
class DataField:
    """Describe one ordered logical Dataset field and its physical unit."""

    name: str
    unit: str


@dataclass(frozen=True, slots=True)
class TransientTemporalContract:
    """Describe authoritative regular-time coordinates at the sample boundary."""

    fields: tuple[DataField, ...]
    tensor_dtype: str
    regular_transition_step: float
    authoritative_source: str
    configured_horizon_source: str
    exact_stop_usage: str

    @property
    def unit(self) -> str:
        """Return the shared unit of every temporal field."""
        units = {field.unit for field in self.fields}
        if len(units) != 1:
            message = "Transient temporal fields must share one physical unit."
            raise RuntimeError(message)
        return next(iter(units))


@dataclass(frozen=True, slots=True)
class TransientSamplingSpec:
    """Select one explicit deterministic transient sample materialization."""

    mode: TransientSampleMode
    rollout_length: int | None = None
    window_stride: int | None = None
    window_offset: int | None = None

    def __post_init__(self) -> None:
        """Reject implicit, irrelevant, or invalid window settings."""
        if self.mode not in TRANSIENT_SAMPLE_MODES:
            available = ", ".join(TRANSIENT_SAMPLE_MODES)
            message = f"Unknown transient sample mode {self.mode!r}. Available modes: {available}."
            raise ValueError(message)
        window_values = (self.rollout_length, self.window_stride, self.window_offset)
        if self.mode == "one_step_transition":
            if any(value is not None for value in window_values):
                message = "one_step_transition does not accept rollout window settings."
                raise ValueError(message)
            return
        for label, value, minimum in (
            ("rollout_length", self.rollout_length, 2),
            ("window_stride", self.window_stride, 1),
            ("window_offset", self.window_offset, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                message = f"{label} must be an integer >= {minimum} for rollout_window."
                raise ValueError(message)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TransientSamplingSpec:
        """Resolve one exact config mapping without sampling defaults."""
        if not isinstance(value, Mapping):
            message = "Transient sampling configuration must be a mapping."
            raise TypeError(message)
        mode = value.get("mode")
        if mode not in TRANSIENT_SAMPLE_MODES:
            available = ", ".join(TRANSIENT_SAMPLE_MODES)
            message = f"Unknown transient sample mode {mode!r}. Available modes: {available}."
            raise ValueError(message)
        required = {"mode"} if mode == "one_step_transition" else {"mode", "rollout_length", "window_stride", "window_offset"}
        if set(value) != required:
            message = f"Transient sampling keys must be exactly {sorted(required)} for mode {mode!r}."
            raise ValueError(message)
        return cls(
            mode=cast("TransientSampleMode", mode),
            rollout_length=value.get("rollout_length"),
            window_stride=value.get("window_stride"),
            window_offset=value.get("window_offset"),
        )

    def as_dict(self) -> dict[str, Any]:
        """Serialize the exact explicit sampling choice."""
        payload: dict[str, Any] = {"mode": self.mode}
        if self.mode == "rollout_window":
            payload.update(
                {
                    "rollout_length": self.rollout_length,
                    "window_stride": self.window_stride,
                    "window_offset": self.window_offset,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class TransientStepContract:
    """Describe the regular-step transient operator sample contract."""

    dynamic_state: tuple[DataField, ...]
    static_spatial_conditioning: tuple[DataField, ...]
    step_boundary_conditioning: tuple[DataField, ...]
    boundary_interval_interpolation: str
    boundary_interval_representation: str
    scalar_conditioning: tuple[DataField, ...]
    temporal: TransientTemporalContract
    target_increments: tuple[DataField, ...]
    archived_ablation_fields: tuple[DataField, ...]
    sampling_modes: tuple[TransientSampleMode, ...]
    tensor_dtype: str
    canonical_storage_representation: str
    target_derivation_stage: str
    material_family_usage: str

    @property
    def time_step(self) -> float:
        """Return the learned transition duration in the temporal unit."""
        return self.temporal.regular_transition_step

    @property
    def time_unit(self) -> str:
        """Return the authoritative temporal unit."""
        return self.temporal.unit


_SOURCE_PROFILE: Final = generation.contracts.get_profile_contract(TRANSIENT_PROFILE_ID)


def _source_field(name: str) -> DataField:
    """Return one Dataset descriptor derived from the Generation source schema."""
    source = _SOURCE_PROFILE.field(name)
    return DataField(source.name, source.unit)


def _scheduled_field(source_name: str, endpoint: str) -> DataField:
    """Return one step-endpoint field derived from a Generation schedule field."""
    source = _SOURCE_PROFILE.field(source_name)
    return DataField(f"{source.name}_{endpoint}", source.unit)


_TIME_UNIT: Final = _SOURCE_PROFILE.field("t").unit
TRANSIENT_STEP_CONTRACT: Final = TransientStepContract(
    dynamic_state=tuple(DataField(field.name, field.unit) for field in _SOURCE_PROFILE.transient_fields),
    static_spatial_conditioning=tuple(
        _source_field(name)
        for name in (
            "x",
            "y",
            "u",
            "v",
            "p",
            "eps_bed",
            "rho_bu_dry",
        )
    ),
    step_boundary_conditioning=(
        _scheduled_field("T_in_bc", "t_n"),
        _scheduled_field("T_in_bc", "t_n_plus_1"),
        _scheduled_field("omega_in_bc", "t_n"),
        _scheduled_field("omega_in_bc", "t_n_plus_1"),
        _source_field("T_amb"),
        DataField("startup_support_time_offset", _TIME_UNIT),
        _scheduled_field("T_in_bc", "startup_support"),
        _scheduled_field("omega_in_bc", "startup_support"),
        DataField("startup_support_present", "1"),
    ),
    boundary_interval_interpolation="linear_between_boundary_schedule_support_nodes",
    boundary_interval_representation=("primitive_regular_endpoints_plus_optional_startup_support_without_extra_training_timestep"),
    scalar_conditioning=tuple(
        _source_field(name)
        for name in (
            "r_surf_0",
            "r_int_surf",
            "f_surf",
            "A_osw",
            "B_osw",
            "C_osw",
            "k_gr",
            "cp_gr_dry",
        )
    ),
    temporal=TransientTemporalContract(
        fields=(
            DataField("t_n", _TIME_UNIT),
            DataField("t_n_plus_1", _TIME_UNIT),
            DataField("dt", _TIME_UNIT),
        ),
        tensor_dtype="float32",
        regular_transition_step=1.0,
        authoritative_source="canonical_hdf5_regular_time_axis",
        configured_horizon_source="generation_scientific_config.time.stop",
        exact_stop_usage="diagnostic_only_no_training_transition_or_rollout",
    ),
    target_increments=tuple(DataField(f"delta_{field.name}", field.unit) for field in _SOURCE_PROFILE.transient_fields),
    archived_ablation_fields=tuple(
        _source_field(name)
        for name in (
            "Kxx",
            "Kxy",
            "Kyy",
            "p_in_bc",
            "X_0_db_field",
        )
    ),
    sampling_modes=TRANSIENT_SAMPLE_MODES,
    tensor_dtype="float32",
    canonical_storage_representation="absolute_physical_states",
    target_derivation_stage="transient_dataset_runtime",
    material_family_usage="metadata_only",
)


def transient_contract_payload() -> dict[str, Any]:
    """Return the exact persisted transient tensor and temporal contract."""
    contract = TRANSIENT_STEP_CONTRACT
    temporal = contract.temporal
    return {
        "state": [{"name": field.name, "unit": field.unit} for field in contract.dynamic_state],
        "static": [{"name": field.name, "unit": field.unit} for field in contract.static_spatial_conditioning],
        "boundary": [{"name": field.name, "unit": field.unit} for field in contract.step_boundary_conditioning],
        "boundary_interval": {
            "interpolation": contract.boundary_interval_interpolation,
            "representation": contract.boundary_interval_representation,
        },
        "scalars": [{"name": field.name, "unit": field.unit} for field in contract.scalar_conditioning],
        "time": {
            "fields": [{"name": field.name, "unit": field.unit} for field in temporal.fields],
            "tensor_dtype": temporal.tensor_dtype,
            "regular_transition_step": {"value": temporal.regular_transition_step, "unit": temporal.unit},
            "authoritative_source": temporal.authoritative_source,
            "configured_horizon_source": temporal.configured_horizon_source,
            "exact_stop_usage": temporal.exact_stop_usage,
        },
        "target": [{"name": field.name, "unit": field.unit} for field in contract.target_increments],
        "sampling": {
            "modes": list(contract.sampling_modes),
            "rollout_target": "ordered_next_state_minus_current_state_sequence",
            "window_boundary": "single_case_regular_transitions_only",
        },
        "tensor_dtype": contract.tensor_dtype,
        "storage": contract.canonical_storage_representation,
        "target_derivation": contract.target_derivation_stage,
        "material_family_usage": contract.material_family_usage,
    }


def transient_contract_digest() -> str:
    """Return the exact path-independent transient Dataset contract digest."""
    payload = {
        "schema_version": TRANSIENT_VIEW_CONTRACT_SCHEMA_VERSION,
        "view": TRANSIENT_VIEW_ID,
        "contract": transient_contract_payload(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
