"""
learning_transient_scaling.py

Fit and admit train-only transient scaling evidence.

Responsibilities:
  - Fit deduplicated absolute-state and zero-preserving increment statistics
  - Persist task, Dataset, membership, profile, and channel identities
  - Encode and decode state, increment, spatial, scalar, and boundary values
  - Produce device-independent serialized evidence and identity digests

Design principles:
  - Statistics consume Train items only and preserve state provenance
  - Increment scaling never centers exact zero increments
  - Persisted evidence is CPU-valued while runtime copies may use one device
  - Constant channels use an explicit unit scale after a positive finite floor

This module does NOT:
  - Load HDF5 or select Dataset memberships
  - Assemble complete model channels or execute model inference
  - Reuse the steady rank-four neuraloperator data processor
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, Literal

import torch

from src import datasets, domain

from .learning_transient_contracts import TransientTensorizerSpec

SCALING_SCHEMA_KIND: Final = "transient_drying_scaling"
SCALING_SCHEMA_VERSION: Final = 1
SCALE_FLOOR: Final = 1.0e-7
_SHA256_LENGTH: Final = 64
_STATE_CHANNELS: Final = 4
_STATIC_CHANNELS: Final = 7
_BOUNDARY_CHANNELS: Final = 9
_SCALAR_CHANNELS: Final = 8
_SPATIAL_BATCH_RANK: Final = 4
_VECTOR_BATCH_RANK: Final = 2
_UNBATCHED_SPATIAL_RANK: Final = 3
TransientScaleMode = Literal["state_std", "delta_rms"]


_TENSOR_NAMES: Final = (
    "state_mean",
    "state_std",
    "delta_rms",
    "increment_scale",
    "static_mean",
    "static_std",
    "scalar_mean",
    "scalar_std",
    "omega_boundary_mean",
    "omega_boundary_std",
)


def _require_sha256(value: Any, *, label: str) -> str:
    """Return one strict lowercase hexadecimal SHA-256 identity."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        message = f"{label} must be a lowercase 64-character SHA-256 digest."
        raise ValueError(message)
    return value


def _canonical_mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return an isolated canonical JSON-compatible mapping."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        message = f"{label} must be JSON-compatible without non-finite values."
        raise ValueError(message) from error
    if not isinstance(decoded, dict) or not decoded:
        message = f"{label} must serialize as a non-empty JSON object."
        raise ValueError(message)
    return decoded


def _concrete_device(value: torch.device | str) -> torch.device:
    """Return one concrete supported runtime device."""
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        message = f"Transient scaling requires CPU or CUDA, got {device}."
        raise ValueError(message)
    if device.type == "cuda" and device.index is None:
        message = "Transient scaling requires an indexed CUDA device."
        raise ValueError(message)
    return device


def _finite_float_tensor(value: Any, *, label: str) -> torch.Tensor:
    """Return an isolated finite contiguous real floating tensor."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"{label} must be one real floating-point tensor."
        raise TypeError(message)
    if not bool(torch.isfinite(value).all().item()):
        message = f"{label} must contain only finite values."
        raise ValueError(message)
    return value.detach().clone().contiguous()


def _population_statistics(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unmodified population mean and standard deviation statistics."""
    return values.mean(dim=1), values.std(dim=1, unbiased=False)


def _population_rms(values: torch.Tensor) -> torch.Tensor:
    """Return the unmodified zero-preserving per-channel population RMS."""
    return torch.sqrt(torch.mean(values.square(), dim=1))


def _operational_scale(values: torch.Tensor, *, floor: float) -> torch.Tensor:
    """Construct a positive normalization scale without changing persisted statistics."""
    return values.clamp_min(floor)


def _channel_view(
    parameter: torch.Tensor,
    value: torch.Tensor,
    *,
    channels: int,
) -> torch.Tensor:
    """Return one per-channel parameter view for BCHW or BLCHW values."""
    if value.ndim not in {4, 5} or value.shape[-3] != channels:
        message = f"Expected rank-four or rank-five tensor with {channels} channels at axis -3, got {tuple(value.shape)}."
        raise ValueError(message)
    shape = [1] * value.ndim
    shape[-3] = channels
    return parameter.view(*shape)


def _validate_runtime_value(
    value: Any,
    *,
    label: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Require one finite runtime tensor on the scaler device and dtype."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"{label} must be one real floating-point tensor."
        raise TypeError(message)
    if not bool(torch.isfinite(value).all().item()):
        message = f"{label} must contain only finite values."
        raise ValueError(message)
    if value.device != device or value.dtype != dtype:
        message = f"{label} must use scaler device/dtype {device}/{dtype}, got {value.device}/{value.dtype}."
        raise ValueError(message)
    return value


@dataclass(frozen=True, slots=True)
class TransientScalingArtifact:
    """
    Persist exact Train-only transient scaling evidence and identity.

    All numeric tensors share one device and dtype. Serialized state always
    emits device-independent JSON-compatible CPU values, so hashes do not
    change when a runtime copy moves between supported devices.

    """

    task_contract_digest: str
    data_contract_digest: str
    tensorizer: TransientTensorizerSpec
    dataset_identity: Mapping[str, Any]
    train_membership_digest: str
    scale_mode: TransientScaleMode
    numerical_floor: float
    unique_train_state_count: int
    unique_transition_count: int
    transition_count: int
    spatial_shape: tuple[int, int]
    state_names: tuple[str, ...]
    static_names: tuple[str, ...]
    boundary_names: tuple[str, ...]
    scalar_names: tuple[str, ...]
    state_mean: torch.Tensor
    state_std: torch.Tensor
    delta_rms: torch.Tensor
    increment_scale: torch.Tensor
    static_mean: torch.Tensor
    static_std: torch.Tensor
    scalar_mean: torch.Tensor
    scalar_std: torch.Tensor
    omega_boundary_mean: torch.Tensor
    omega_boundary_std: torch.Tensor
    horizon: float

    def __post_init__(self) -> None:
        """Isolate values and reject identity, field, shape, or device drift."""
        self._validate_identity()
        self._validate_fields()
        self._validate_counts_and_horizon()
        self._validate_spatial_shape()
        self._validate_tensors()

    def _validate_identity(self) -> None:
        """Validate task, profile, Dataset, and membership identities."""
        task = domain.tasks.registry.get_task("transient_drying")
        if (
            _require_sha256(
                self.task_contract_digest,
                label="task_contract_digest",
            )
            != task.contract_digest
        ):
            message = "Scaling task contract does not match transient_drying."
            raise ValueError(message)
        if (
            _require_sha256(
                self.data_contract_digest,
                label="data_contract_digest",
            )
            != task.data_contract_digest
        ):
            message = "Scaling data contract does not match transient_drying."
            raise ValueError(message)
        if not isinstance(self.tensorizer, TransientTensorizerSpec):
            message = "tensorizer must be a TransientTensorizerSpec."
            raise TypeError(message)
        identity = _canonical_mapping(
            self.dataset_identity,
            label="dataset_identity",
        )
        object.__setattr__(self, "dataset_identity", identity)
        _require_sha256(
            self.train_membership_digest,
            label="train_membership_digest",
        )
        if self.scale_mode not in {"state_std", "delta_rms"}:
            message = "scale_mode must be 'state_std' or 'delta_rms'."
            raise ValueError(message)
        if (
            not isinstance(self.numerical_floor, (int, float))
            or isinstance(self.numerical_floor, bool)
            or not math.isfinite(float(self.numerical_floor))
            or float(self.numerical_floor) <= 0.0
        ):
            message = "numerical_floor must be one positive finite number."
            raise ValueError(message)
        if float(self.numerical_floor) != SCALE_FLOOR:
            message = f"numerical_floor must equal the current scaling floor {SCALE_FLOOR}."
            raise ValueError(message)
        object.__setattr__(self, "numerical_floor", float(self.numerical_floor))

    def _validate_fields(self) -> None:
        """Bind every persisted field group to the Dataset contract."""
        contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
        expected = (
            tuple(field.name for field in contract.dynamic_state),
            tuple(field.name for field in contract.static_spatial_conditioning),
            tuple(field.name for field in contract.step_boundary_conditioning),
            tuple(field.name for field in contract.scalar_conditioning),
        )
        observed = (
            self.state_names,
            self.static_names,
            self.boundary_names,
            self.scalar_names,
        )
        if observed != expected:
            message = "Transient scaling field names disagree with TRANSIENT_STEP_CONTRACT."
            raise ValueError(message)

    def _validate_counts_and_horizon(self) -> None:
        """Validate positive exact counts and the configured time horizon."""
        for label, value in (
            ("unique_train_state_count", self.unique_train_state_count),
            ("unique_transition_count", self.unique_transition_count),
            ("transition_count", self.transition_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                message = f"{label} must be a positive integer."
                raise ValueError(message)
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, (int, float))
            or not math.isfinite(float(self.horizon))
            or float(self.horizon) <= 0.0
        ):
            message = "horizon must be one positive finite number."
            raise ValueError(message)
        object.__setattr__(self, "horizon", float(self.horizon))

    def _validate_spatial_shape(self) -> None:
        """Require the exact positive Train-admitted [Y, X] spatial shape."""
        shape = self.spatial_shape
        if not isinstance(shape, tuple) or len(shape) != _SPATIAL_BATCH_RANK - _VECTOR_BATCH_RANK:
            message = "spatial_shape must be one exact (Y, X) tuple."
            raise ValueError(message)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in shape):
            message = "spatial_shape must contain two positive integer axes [Y, X]."
            raise ValueError(message)

    def _validate_tensors(self) -> None:
        """Validate shapes, shared placement, and positive fitted scales."""
        expected_shapes = {
            "state_mean": (_STATE_CHANNELS,),
            "state_std": (_STATE_CHANNELS,),
            "delta_rms": (_STATE_CHANNELS,),
            "increment_scale": (_STATE_CHANNELS,),
            "static_mean": (_STATIC_CHANNELS,),
            "static_std": (_STATIC_CHANNELS,),
            "scalar_mean": (_SCALAR_CHANNELS,),
            "scalar_std": (_SCALAR_CHANNELS,),
            "omega_boundary_mean": (),
            "omega_boundary_std": (),
        }
        reference_device: torch.device | None = None
        reference_dtype: torch.dtype | None = None
        for name, shape in expected_shapes.items():
            tensor = _finite_float_tensor(getattr(self, name), label=name)
            if tuple(tensor.shape) != shape:
                message = f"{name} must have shape {shape}, got {tuple(tensor.shape)}."
                raise ValueError(message)
            if name in {"state_std", "delta_rms"} and not bool((tensor >= 0).all().item()):
                message = f"{name} must contain only non-negative values."
                raise ValueError(message)
            if name not in {"state_std", "delta_rms"} and name.endswith(("std", "rms", "scale")) and not bool((tensor > 0).all().item()):
                message = f"{name} must contain only positive values."
                raise ValueError(message)
            if reference_device is None:
                reference_device = tensor.device
                reference_dtype = tensor.dtype
            elif tensor.device != reference_device or tensor.dtype != reference_dtype:
                message = "All transient scaling tensors must share device and dtype."
                raise ValueError(message)
            object.__setattr__(self, name, tensor)
        selected = self.state_std if self.scale_mode == "state_std" else self.delta_rms
        expected_increment_scale = _operational_scale(selected, floor=self.numerical_floor)
        if not torch.equal(self.increment_scale, expected_increment_scale):
            message = "increment_scale disagrees with persisted scale_mode statistics and numerical_floor."
            raise ValueError(message)

    @property
    def device(self) -> torch.device:
        """Return the shared runtime tensor device."""
        return self.state_mean.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the shared runtime tensor dtype."""
        return self.state_mean.dtype

    @property
    def digest(self) -> str:
        """Return the device-independent SHA-256 of all persisted evidence."""
        encoded = json.dumps(
            self.state_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact identity and detached CPU numeric evidence."""
        tensor_values = {name: getattr(self, name).detach().to(device="cpu").tolist() for name in _TENSOR_NAMES}
        return {
            "schema_kind": SCALING_SCHEMA_KIND,
            "schema_version": SCALING_SCHEMA_VERSION,
            "task_contract_digest": self.task_contract_digest,
            "data_contract_digest": self.data_contract_digest,
            "tensorizer": self.tensorizer.as_dict(),
            "dataset_identity": dict(self.dataset_identity),
            "train_membership_digest": self.train_membership_digest,
            "scale_mode": self.scale_mode,
            "numerical_floor": self.numerical_floor,
            "unique_train_state_count": self.unique_train_state_count,
            "unique_transition_count": self.unique_transition_count,
            "transition_count": self.transition_count,
            "spatial_shape": list(self.spatial_shape),
            "state_names": list(self.state_names),
            "static_names": list(self.static_names),
            "boundary_names": list(self.boundary_names),
            "scalar_names": list(self.scalar_names),
            "horizon": self.horizon,
            **tensor_values,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> TransientScalingArtifact:
        """Strictly admit one JSON-compatible scaling artifact state."""
        if not isinstance(state, Mapping):
            message = "Transient scaling state must be a mapping."
            raise TypeError(message)
        identity_keys = {
            "schema_kind",
            "schema_version",
            "task_contract_digest",
            "data_contract_digest",
            "tensorizer",
            "dataset_identity",
            "train_membership_digest",
            "scale_mode",
            "numerical_floor",
            "unique_train_state_count",
            "unique_transition_count",
            "transition_count",
            "spatial_shape",
            "state_names",
            "static_names",
            "boundary_names",
            "scalar_names",
            "horizon",
        }
        required = identity_keys.union(_TENSOR_NAMES)
        if set(state) != required:
            message = "Transient scaling state keys do not match the current schema."
            raise ValueError(message)
        if state["schema_kind"] != SCALING_SCHEMA_KIND or state["schema_version"] != SCALING_SCHEMA_VERSION:
            message = "Transient scaling state has an unsupported schema."
            raise ValueError(message)
        raw_tensorizer = state["tensorizer"]
        if not isinstance(raw_tensorizer, Mapping):
            message = "Saved tensorizer identity must be a mapping."
            raise TypeError(message)
        selection = {
            "input_profile": raw_tensorizer.get("input_profile"),
            "temporal_conditioning": raw_tensorizer.get("temporal_conditioning"),
        }
        tensorizer = TransientTensorizerSpec.from_mapping(selection)
        if dict(raw_tensorizer) != tensorizer.as_dict():
            message = "Saved tensorizer identity does not match its resolved contract."
            raise ValueError(message)
        tensors = {name: torch.as_tensor(state[name], dtype=torch.float32) for name in _TENSOR_NAMES}
        return cls(
            task_contract_digest=state["task_contract_digest"],
            data_contract_digest=state["data_contract_digest"],
            tensorizer=tensorizer,
            dataset_identity=state["dataset_identity"],
            train_membership_digest=state["train_membership_digest"],
            scale_mode=state["scale_mode"],
            numerical_floor=state["numerical_floor"],
            unique_train_state_count=state["unique_train_state_count"],
            unique_transition_count=state["unique_transition_count"],
            transition_count=state["transition_count"],
            spatial_shape=tuple(state["spatial_shape"]),
            state_names=tuple(state["state_names"]),
            static_names=tuple(state["static_names"]),
            boundary_names=tuple(state["boundary_names"]),
            scalar_names=tuple(state["scalar_names"]),
            horizon=state["horizon"],
            **tensors,
        )

    def to(self, device: torch.device | str) -> TransientScalingArtifact:
        """Return an equivalent runtime artifact on one concrete device."""
        target = _concrete_device(device)
        changes = {name: getattr(self, name).to(device=target) for name in _TENSOR_NAMES}
        return replace(self, **changes)

    def encode_state(self, value: torch.Tensor) -> torch.Tensor:
        """Standardize physical absolute-state channels."""
        tensor = _validate_runtime_value(
            value,
            label="state",
            device=self.device,
            dtype=self.dtype,
        )
        mean = _channel_view(
            self.state_mean,
            tensor,
            channels=_STATE_CHANNELS,
        )
        scale = _channel_view(
            _operational_scale(self.state_std, floor=self.numerical_floor),
            tensor,
            channels=_STATE_CHANNELS,
        )
        return (tensor - mean) / scale

    def decode_state(self, value: torch.Tensor) -> torch.Tensor:
        """Decode standardized absolute-state channels."""
        tensor = _validate_runtime_value(
            value,
            label="encoded state",
            device=self.device,
            dtype=self.dtype,
        )
        mean = _channel_view(
            self.state_mean,
            tensor,
            channels=_STATE_CHANNELS,
        )
        scale = _channel_view(
            _operational_scale(self.state_std, floor=self.numerical_floor),
            tensor,
            channels=_STATE_CHANNELS,
        )
        return tensor * scale + mean

    def encode_delta(self, value: torch.Tensor) -> torch.Tensor:
        """Scale increments without centering so exact zeros remain zero."""
        tensor = _validate_runtime_value(
            value,
            label="increment",
            device=self.device,
            dtype=self.dtype,
        )
        scale = _channel_view(
            self.increment_scale,
            tensor,
            channels=_STATE_CHANNELS,
        )
        return tensor / scale

    def decode_delta(self, value: torch.Tensor) -> torch.Tensor:
        """Decode zero-preserving scaled increments."""
        tensor = _validate_runtime_value(
            value,
            label="scaled increment",
            device=self.device,
            dtype=self.dtype,
        )
        scale = _channel_view(
            self.increment_scale,
            tensor,
            channels=_STATE_CHANNELS,
        )
        decoded = tensor * scale
        if not bool(torch.isfinite(decoded).all().item()):
            message = "Decoded transient increment is non-finite."
            raise FloatingPointError(message)
        return decoded

    def encode_static(self, value: torch.Tensor) -> torch.Tensor:
        """Standardize static spatial conditioning channels."""
        tensor = _validate_runtime_value(
            value,
            label="static conditioning",
            device=self.device,
            dtype=self.dtype,
        )
        if tensor.ndim != _SPATIAL_BATCH_RANK or tensor.shape[1] != _STATIC_CHANNELS:
            message = "Static conditioning must have shape [B,7,Y,X]."
            raise ValueError(message)
        return (tensor - self.static_mean.view(1, _STATIC_CHANNELS, 1, 1)) / self.static_std.view(1, _STATIC_CHANNELS, 1, 1)

    def encode_scalars(self, value: torch.Tensor) -> torch.Tensor:
        """Standardize scalar material-conditioning channels."""
        tensor = _validate_runtime_value(
            value,
            label="scalar conditioning",
            device=self.device,
            dtype=self.dtype,
        )
        if tensor.ndim != _VECTOR_BATCH_RANK or tensor.shape[1] != _SCALAR_CHANNELS:
            message = "Scalar conditioning must have shape [B,8]."
            raise ValueError(message)
        return (tensor - self.scalar_mean.view(1, _SCALAR_CHANNELS)) / self.scalar_std.view(1, _SCALAR_CHANNELS)

    def encode_boundary(self, value: torch.Tensor) -> torch.Tensor:
        """
        Scale interval conditioning and neutralize absent startup evidence.

        Regular and startup temperature values share the Train-state temperature
        scale. Regular and present startup humidity-ratio values share one fitted
        omega scale. The startup flag remains exactly binary and unstandardized.
        """
        tensor = _validate_runtime_value(
            value,
            label="boundary conditioning",
            device=self.device,
            dtype=self.dtype,
        )
        if tensor.ndim != _VECTOR_BATCH_RANK or tensor.shape[1] != _BOUNDARY_CHANNELS:
            message = "Boundary conditioning must have shape [B,9]."
            raise ValueError(message)
        flag = tensor[:, 8]
        if not bool(((flag == 0) | (flag == 1)).all().item()):
            message = "startup_support_present must contain exact binary values."
            raise ValueError(message)
        encoded = tensor.clone()
        temperature_scale = _operational_scale(self.state_std, floor=self.numerical_floor)[0]
        encoded[:, 0:2] = (encoded[:, 0:2] - self.state_mean[0]) / temperature_scale
        encoded[:, 2:4] = (encoded[:, 2:4] - self.omega_boundary_mean) / self.omega_boundary_std
        encoded[:, 4] = (encoded[:, 4] - self.state_mean[0]) / temperature_scale
        encoded[:, 5] = encoded[:, 5] / self.horizon
        encoded[:, 6] = (encoded[:, 6] - self.state_mean[0]) / temperature_scale
        encoded[:, 7] = (encoded[:, 7] - self.omega_boundary_mean) / self.omega_boundary_std
        present = flag == 1
        encoded[:, 5:8] = torch.where(
            present[:, None],
            encoded[:, 5:8],
            torch.zeros_like(encoded[:, 5:8]),
        )
        encoded[:, 8] = flag
        return encoded


def _raw_fit_tensor(
    item: Mapping[str, Any],
    name: str,
) -> torch.Tensor:
    """Return one detached finite float32 CPU tensor from a Train item."""
    value = item.get(name)
    tensor = _finite_float_tensor(value, label=name)
    return tensor.to(device="cpu", dtype=torch.float32)


def _scalar_time(
    time: Mapping[str, Any],
    name: str,
) -> float:
    """Return one finite scalar temporal value."""
    tensor = _finite_float_tensor(time.get(name), label=f"time.{name}")
    if tensor.numel() != 1:
        message = f"time.{name} must contain one scalar value."
        raise ValueError(message)
    return float(tensor.detach().cpu().item())


def _metadata_identity(
    metadata: Mapping[str, Any],
) -> tuple[str, int, int]:
    """Return strict simulation and transition-index identity."""
    simulation = metadata.get("simulation_case_id")
    index_n = metadata.get("time_index_n")
    index_next = metadata.get("time_index_n_plus_1")
    if not isinstance(simulation, str) or not simulation:
        message = "simulation_case_id must be one non-empty string."
        raise ValueError(message)
    if isinstance(index_n, bool) or not isinstance(index_n, int) or index_n < 0:
        message = "time_index_n must be a non-negative integer."
        raise ValueError(message)
    if isinstance(index_next, bool) or not isinstance(index_next, int) or index_next < 0:
        message = "time_index_n_plus_1 must be a non-negative integer."
        raise ValueError(message)
    if index_next <= index_n:
        message = "Transient time indexes must advance."
        raise ValueError(message)
    return simulation, index_n, index_next


def _validate_fit_shapes(
    *,
    state: torch.Tensor,
    target: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    scalars: torch.Tensor,
    expected_spatial_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    """Require exact unbatched one-step Train item shapes."""
    if state.ndim != _UNBATCHED_SPATIAL_RANK or state.shape[0] != _STATE_CHANNELS:
        message = "Train state must have shape [4,Y,X]."
        raise ValueError(message)
    spatial_shape = (int(state.shape[1]), int(state.shape[2]))
    if min(spatial_shape) < 1:
        message = "Train state spatial axes must be non-empty."
        raise ValueError(message)
    expected = {
        "target": (_STATE_CHANNELS, *spatial_shape),
        "static": (_STATIC_CHANNELS, *spatial_shape),
        "boundary": (_BOUNDARY_CHANNELS,),
        "scalars": (_SCALAR_CHANNELS,),
    }
    for label, tensor in (
        ("target", target),
        ("static", static),
        ("boundary", boundary),
        ("scalars", scalars),
    ):
        if tuple(tensor.shape) != expected[label]:
            message = f"Train {label} must have shape {expected[label]}, got {tuple(tensor.shape)}."
            raise ValueError(message)
    if expected_spatial_shape is not None and spatial_shape != expected_spatial_shape:
        message = "Train items do not share one spatial shape."
        raise ValueError(message)
    flag = boundary[8]
    if float(flag.item()) not in {0.0, 1.0}:
        message = "startup_support_present must be exactly zero or one."
        raise ValueError(message)
    return spatial_shape


def _positive_horizon(value: Any) -> float:
    """Return one positive finite configured horizon."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        message = "horizon must be one positive finite number."
        raise ValueError(message)
    return float(value)


def fit_transient_scaling(
    items: Iterable[Mapping[str, Any]],
    *,
    tensorizer: TransientTensorizerSpec,
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    horizon: float,
    scale_mode: TransientScaleMode = "state_std",
) -> TransientScalingArtifact:
    """
    Fit task-specific statistics from raw one-step Train items only.

    Absolute states are deduplicated by simulation and time index. Repeated
    evidence for one state must agree exactly. Validation/OOD data are not an
    argument and therefore cannot influence fitted statistics.
    """
    resolved_horizon = _positive_horizon(horizon)
    if scale_mode not in {"state_std", "delta_rms"}:
        message = "scale_mode must be 'state_std' or 'delta_rms'."
        raise ValueError(message)
    states: dict[tuple[str, int], torch.Tensor] = {}
    deltas: list[torch.Tensor] = []
    statics: list[torch.Tensor] = []
    scalars: list[torch.Tensor] = []
    omega_values: list[torch.Tensor] = []
    transitions: dict[tuple[str, int, int], torch.Tensor] = {}
    transition_count = 0
    spatial_shape: tuple[int, int] | None = None

    for item in items:
        if not isinstance(item, Mapping):
            message = "Transient scaling items must be mappings."
            raise TypeError(message)
        state = _raw_fit_tensor(item, "state")
        target = _raw_fit_tensor(item, "target")
        static = _raw_fit_tensor(item, "static")
        boundary = _raw_fit_tensor(item, "boundary")
        scalar_values = _raw_fit_tensor(item, "scalars")
        spatial_shape = _validate_fit_shapes(
            state=state,
            target=target,
            static=static,
            boundary=boundary,
            scalars=scalar_values,
            expected_spatial_shape=spatial_shape,
        )
        metadata = item.get("metadata")
        time = item.get("time")
        if not isinstance(metadata, Mapping) or not isinstance(time, Mapping):
            message = "Train items require metadata and time mappings."
            raise TypeError(message)
        simulation, index_n, index_next = _metadata_identity(metadata)
        t_n = _scalar_time(time, "t_n")
        t_next = _scalar_time(time, "t_n_plus_1")
        dt = _scalar_time(time, "dt")
        if not math.isclose(dt, 1.0, rel_tol=0.0, abs_tol=1.0e-6) or not math.isclose(
            t_next - t_n,
            dt,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            message = "Transient scaling requires fixed regular 1h transitions."
            raise ValueError(message)
        if t_n < 0.0 or t_next > resolved_horizon + 1.0e-6:
            message = "Train time evidence lies outside the configured horizon."
            raise ValueError(message)

        current_key = (simulation, index_n)
        next_key = (simulation, index_next)
        for key, state_value in (
            (current_key, state),
            (next_key, state + target),
        ):
            prior = states.get(key)
            if prior is not None and not torch.equal(prior, state_value):
                message = "Duplicate transient state evidence disagrees exactly."
                raise ValueError(message)
            states[key] = state_value

        transition_key = (simulation, index_n, index_next)
        prior_transition = transitions.get(transition_key)
        if prior_transition is not None and not torch.equal(prior_transition, target):
            message = "Duplicate transient transition evidence disagrees exactly."
            raise ValueError(message)
        if prior_transition is None:
            transitions[transition_key] = target
            deltas.append(target)
        statics.append(static)
        scalars.append(scalar_values)
        omega_values.append(boundary[2:4])
        if float(boundary[8].item()) == 1.0:
            omega_values.append(boundary[7:8])
        transition_count += 1

    if transition_count == 0 or spatial_shape is None:
        message = "Transient scaling fit requires at least one Train item."
        raise ValueError(message)

    state_matrix = torch.stack(tuple(states.values())).permute(1, 0, 2, 3).reshape(_STATE_CHANNELS, -1)
    delta_matrix = torch.stack(deltas).permute(1, 0, 2, 3).reshape(_STATE_CHANNELS, -1)
    static_matrix = torch.stack(statics).permute(1, 0, 2, 3).reshape(_STATIC_CHANNELS, -1)
    scalar_matrix = torch.stack(scalars).transpose(0, 1)
    omega_matrix = torch.cat(omega_values).reshape(1, -1)

    state_mean, state_std = _population_statistics(state_matrix)
    delta_rms = _population_rms(delta_matrix)
    selected_increment_statistic = state_std if scale_mode == "state_std" else delta_rms
    increment_scale = _operational_scale(selected_increment_statistic, floor=SCALE_FLOOR)
    static_mean, static_std = _population_statistics(static_matrix)
    scalar_mean, scalar_std = _population_statistics(scalar_matrix)
    omega_mean, omega_std = _population_statistics(omega_matrix)
    static_std = _operational_scale(static_std, floor=SCALE_FLOOR)
    scalar_std = _operational_scale(scalar_std, floor=SCALE_FLOOR)
    omega_std = _operational_scale(omega_std, floor=SCALE_FLOOR)

    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    task = domain.tasks.registry.get_task("transient_drying")
    return TransientScalingArtifact(
        task_contract_digest=task.contract_digest,
        data_contract_digest=task.data_contract_digest,
        tensorizer=tensorizer,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        scale_mode=scale_mode,
        numerical_floor=SCALE_FLOOR,
        unique_train_state_count=len(states),
        unique_transition_count=len(transitions),
        transition_count=transition_count,
        spatial_shape=spatial_shape,
        state_names=tuple(field.name for field in contract.dynamic_state),
        static_names=tuple(field.name for field in contract.static_spatial_conditioning),
        boundary_names=tuple(field.name for field in contract.step_boundary_conditioning),
        scalar_names=tuple(field.name for field in contract.scalar_conditioning),
        state_mean=state_mean,
        state_std=state_std,
        delta_rms=delta_rms,
        increment_scale=increment_scale,
        static_mean=static_mean,
        static_std=static_std,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        omega_boundary_mean=omega_mean.squeeze(0),
        omega_boundary_std=omega_std.squeeze(0),
        horizon=resolved_horizon,
    )
