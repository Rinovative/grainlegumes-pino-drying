"""Own the one Train-fitted channel scaling of both tasks: case-equal streaming statistics, persisted state and encodings.

Responsibilities:
  - Accumulate case-equal population moments without retaining Train tensors
  - Fit the steady scaling from Train cases, one case at a time
  - Hold, validate, serialize and digest the one persisted scaling state of both tasks
  - Encode and decode states, increments, static fields, scalars, boundary values and steady model inputs with the Train scale
  - Nondimensionalize the steady representation by each case's mean prescribed inlet pressure P
  - State the preprocessing specification that the shared scaling identity binds, and admit a saved state under that identity

Design principles:
  - Every Train case has weight one in the fit; the values within one case weigh equally
  - The scale of a channel is max(sigma, 1e-7) of its population standard deviation sigma; the state persists sigma itself
  - A boundary value of a state uses that state's mean and scale: steady ``p_in_bc`` the scale of ``p``, the transient
    inlet and ambient temperatures the scale of ``T``
  - The steady network sees p_in_bc/P and learns p/P, u/P, v/P; P is never a model input, every steady encode and decode
    takes it explicitly, and the physical state statistics stay available for metrics and selection
  - Increments use the state scale without centering, so exact zeros stay zero
  - Persisted values are device-independent JSON-compatible lists, so digests do not depend on the runtime device
  - A state of another schema, such as a neuralop normalizer or an earlier transient scaling, is refused, never converted

This module does NOT:
  - Choose split membership, load Datasets or construct DataLoaders
  - Validate the ordered transient Train chain; the transient fit owns that and builds this state
  - Cache, envelope or persist scaling artifacts; the shared scaling service owns that

Notes:
  Steady representation, P the mean of p_in_bc over the domain's inlet mask (domain.physics.brinkman):
    encoded input p_in_bc = (p_in_bc/P - mean_p)/scale_p, encoded state = (y/P - mean)/scale, decoding the exact inverse.
    The Darcy-Brinkman equations are linear with homogeneous other boundary conditions and outlet gauge 0, so the
    physical (p, u, v) scale exactly with the inlet pressure and the dimensionless fields do not depend on its level.
  Channel groups:
    - steady_flow: ``state`` (the dimensionless outputs p/P, u/P, v/P that the encoding standardizes), ``static``
      (every input except ``p_in_bc``) and ``physical_state`` (the physical outputs, for metrics and selection)
    - transient_drying: ``state``, ``static``, ``scalar``, ``omega_boundary`` (inlet humidity ratio) and
      ``grain_moisture`` (the derived w_gr, used only by diagnostics)

"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Final, cast

import torch

from src import domain
from src.datasets.contracts import dataset_contracts_transient as transient_contract

SCALING_SCHEMA_KIND: Final = "train_channel_scaling"
PREPROCESSING_KIND: Final = domain.tasks.spec.TRAIN_SCALING_KIND
SCALE_FLOOR: Final = 1.0e-7
STEADY_TASK: Final = "steady_flow"
TRANSIENT_TASK: Final = "transient_drying"
STEADY_BOUNDARY_STATES: Final = MappingProxyType({"p_in_bc": "p"})
# The steady preprocessing names its per-case state reference, so a state of another representation is refused.
STEADY_STATE_REFERENCE: Final = "mean_inlet_pressure"
_STATE_KEYS: Final = frozenset(
    {
        "schema_kind",
        "task",
        "task_contract_digest",
        "data_contract_digest",
        "preprocessing",
        "dataset_identity",
        "train_membership_digest",
        "scale_floor",
        "train_case_count",
        "spatial_shape",
        "horizon",
        "groups",
    }
)
_GROUP_KEYS: Final = frozenset({"names", "mean", "std"})
_SHA256_LENGTH: Final = 64
_SPATIAL_RANKS: Final = frozenset({4, 5})
_SPATIAL_BATCH_RANK: Final = 4
_VECTOR_BATCH_RANK: Final = 2
_UNBATCHED_SPATIAL_RANK: Final = 3
_SPATIAL_AXES: Final = 2
_BOUNDARY_CHANNELS: Final = 9


def expected_groups(task: str) -> dict[str, tuple[str, ...]]:
    """Return the ordered channel names of every scaling group of one task."""
    if task == STEADY_TASK:
        spec = domain.tasks.registry.get_task(STEADY_TASK)
        return {
            "state": spec.output_names,
            "static": tuple(name for name in spec.input_names if name not in STEADY_BOUNDARY_STATES),
            "physical_state": spec.output_names,
        }
    if task == TRANSIENT_TASK:
        contract = transient_contract.TRANSIENT_STEP_CONTRACT
        return {
            "state": tuple(field.name for field in contract.dynamic_state),
            "static": tuple(field.name for field in contract.static_spatial_conditioning),
            "scalar": tuple(field.name for field in contract.scalar_conditioning),
            "omega_boundary": ("omega_in_bc",),
            "grain_moisture": ("w_gr",),
        }
    message = f"Train scaling is defined for {STEADY_TASK!r} and {TRANSIENT_TASK!r}, not {task!r}."
    raise ValueError(message)


def _require_sha256(value: Any, *, label: str) -> str:
    """Return one strict lowercase hexadecimal SHA-256 identity."""
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        message = f"{label} must be a lowercase 64-character SHA-256 digest."
        raise ValueError(message)
    return value


def _canonical_mapping(value: Any, *, label: str) -> dict[str, Any]:
    """Return an isolated canonical JSON-compatible non-empty mapping."""
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping."
        raise TypeError(message)
    try:
        decoded = json.loads(json.dumps(dict(value), ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True))
    except (TypeError, ValueError) as error:
        message = f"{label} must be JSON-compatible without non-finite values."
        raise ValueError(message) from error
    if not isinstance(decoded, dict) or not decoded:
        message = f"{label} must serialize as a non-empty JSON object."
        raise ValueError(message)
    return decoded


def _finite_float_tensor(value: Any, *, label: str) -> torch.Tensor:
    """Return an isolated finite contiguous real floating tensor."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"{label} must be one real floating-point tensor."
        raise TypeError(message)
    if not bool(torch.isfinite(value).all().item()):
        message = f"{label} must contain only finite values."
        raise ValueError(message)
    return value.detach().clone().contiguous()


def _concrete_device(value: torch.device | str) -> torch.device:
    """Return one concrete supported runtime device."""
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        message = f"Train scaling requires CPU or CUDA, got {device}."
        raise ValueError(message)
    if device.type == "cuda" and device.index is None:
        message = "Train scaling requires an indexed CUDA device."
        raise ValueError(message)
    return device


def train_item_tensor(item: Mapping[str, Any], name: str) -> torch.Tensor:
    """Return one detached finite contiguous float32 CPU copy of a Train item tensor."""
    value = item.get(name)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"Train item {name} must be one real floating-point tensor."
        raise TypeError(message)
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    if not bool(torch.isfinite(tensor).all().item()):
        message = f"Train item {name} must contain only finite values."
        raise ValueError(message)
    return tensor


def _mean_inlet_pressure(inputs: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
    """Return the detached steady reference P ``[B]`` of physical inputs ``[B, C, Y, X]``, refusing a non-positive one."""
    names = domain.tasks.registry.get_task(STEADY_TASK).input_names
    with torch.no_grad():
        reference = domain.physics.brinkman.mean_inlet_pressure(inputs, input_fields=names)
    invalid = ~(torch.isfinite(reference) & (reference > 0))
    if bool(invalid.any().item()):
        position = int(invalid.nonzero()[0, 0].item())
        message = (
            f"Steady case {offset + position} has mean prescribed inlet pressure {float(reference[position])}; "
            "the steady representation divides by it, so it must be finite and positive."
        )
        raise ValueError(message)
    return reference


class CaseWeightedMoments:
    """
    Stream population moments in which every case weighs one and the values within one case weigh equally.

    Values join the open case through an exact parallel merge; :meth:`close_case` merges the open case's
    mean and population variance into the running moments with weight one. The result equals the mean
    and population variance over all values weighted by ``1 / (cases * values of their case)``.
    """

    def __init__(self, channels: int) -> None:
        """Initialize empty float64 channel moments."""
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
            message = "Case-weighted moment channels must be a positive integer."
            raise ValueError(message)
        self._channels = channels
        self._case_count = 0
        self._mean = torch.zeros(channels, dtype=torch.float64)
        self._m2 = torch.zeros(channels, dtype=torch.float64)
        self._open_count = 0
        self._open_mean = torch.zeros(channels, dtype=torch.float64)
        self._open_m2 = torch.zeros(channels, dtype=torch.float64)

    @property
    def case_count(self) -> int:
        """Return the number of closed cases."""
        return self._case_count

    def update(self, value: torch.Tensor) -> None:
        """Merge one finite CPU tensor whose leading axis is channel into the open case."""
        if value.ndim < 1 or value.shape[0] != self._channels:
            message = f"Case-weighted moments require a leading channel axis of length {self._channels}."
            raise ValueError(message)
        flattened = value.reshape(self._channels, -1).to(dtype=torch.float64)
        chunk_count = int(flattened.shape[1])
        if chunk_count == 0:
            return
        chunk_variance, chunk_mean = torch.var_mean(flattened, dim=1, correction=0)
        total = self._open_count + chunk_count
        delta = chunk_mean - self._open_mean
        self._open_mean = self._open_mean + delta * (chunk_count / total)
        self._open_m2 = self._open_m2 + chunk_variance * chunk_count + delta.square() * (self._open_count * chunk_count / total)
        self._open_count = total

    def close_case(self) -> None:
        """Merge the open case with weight one and start the next case."""
        if self._open_count == 0:
            message = "A case must contribute at least one value before it closes."
            raise ValueError(message)
        case_variance = self._open_m2 / self._open_count
        self._case_count += 1
        delta = self._open_mean - self._mean
        self._mean = self._mean + delta / self._case_count
        self._m2 = self._m2 + case_variance + delta.square() * ((self._case_count - 1) / self._case_count)
        self._open_count = 0
        self._open_mean = torch.zeros(self._channels, dtype=torch.float64)
        self._open_m2 = torch.zeros(self._channels, dtype=torch.float64)

    def statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return float32 case-equal population mean and standard deviation vectors."""
        if self._case_count < 1 or self._open_count:
            message = "Case-weighted statistics require at least one closed case and no open one."
            raise ValueError(message)
        variance = (self._m2 / self._case_count).clamp_min(0.0)
        return self._mean.to(dtype=torch.float32), torch.sqrt(variance).to(dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class ChannelStatistics:
    """Hold the ordered names, Train mean and population standard deviation of one channel group."""

    names: tuple[str, ...]
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        """Isolate the tensors and require one finite value per name and a non-negative standard deviation."""
        if not isinstance(self.names, tuple) or not self.names or len(set(self.names)) != len(self.names):
            message = "Channel names must be a non-empty tuple of unique names."
            raise ValueError(message)
        if any(not isinstance(name, str) or not name for name in self.names):
            message = "Channel names must be non-empty strings."
            raise ValueError(message)
        mean = _finite_float_tensor(self.mean, label="mean")
        std = _finite_float_tensor(self.std, label="std")
        if tuple(mean.shape) != (len(self.names),) or tuple(std.shape) != (len(self.names),):
            message = f"Channel statistics of {list(self.names)} must have shape ({len(self.names)},)."
            raise ValueError(message)
        if mean.device != std.device or mean.dtype != std.dtype:
            message = "Channel mean and std must share device and dtype."
            raise ValueError(message)
        if not bool((std >= 0).all().item()):
            message = "Channel standard deviations must be non-negative."
            raise ValueError(message)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def scale(self) -> torch.Tensor:
        """Return the operational scale max(sigma, floor)."""
        return self.std.clamp_min(SCALE_FLOOR)

    def to(self, device: torch.device) -> ChannelStatistics:
        """Return the same statistics on one device."""
        return ChannelStatistics(names=self.names, mean=self.mean.to(device=device), std=self.std.to(device=device))

    def as_dict(self) -> dict[str, Any]:
        """Serialize names and CPU values."""
        return {"names": list(self.names), "mean": self.mean.detach().cpu().tolist(), "std": self.std.detach().cpu().tolist()}


def _channel_view(parameter: torch.Tensor, value: torch.Tensor, *, channels: int) -> torch.Tensor:
    """Return one per-channel parameter view for BCHW or BLCHW values."""
    if value.ndim not in _SPATIAL_RANKS or value.shape[-3] != channels:
        message = f"Expected rank-four or rank-five tensor with {channels} channels at axis -3, got {tuple(value.shape)}."
        raise ValueError(message)
    shape = [1] * value.ndim
    shape[-3] = channels
    return parameter.view(*shape)


def _runtime_value(value: Any, *, label: str, device: torch.device, dtype: torch.dtype, require_finite: bool = True) -> torch.Tensor:
    """Require one real runtime tensor on the scaling's device and dtype, finite unless diagnostics keep IEEE values."""
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.is_complex():
        message = f"{label} must be one real floating-point tensor."
        raise TypeError(message)
    if require_finite and not bool(torch.isfinite(value).all().item()):
        message = f"{label} must contain only finite values."
        raise ValueError(message)
    if value.device != device or value.dtype != dtype:
        message = f"{label} must use scaling device/dtype {device}/{dtype}, got {value.device}/{value.dtype}."
        raise ValueError(message)
    return value


def _positive_shape(value: Any, *, label: str) -> tuple[int, int]:
    """Return one exact positive ``(Y, X)`` shape."""
    if (
        not isinstance(value, tuple)
        or len(value) != _SPATIAL_AXES
        or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < 1 for axis in value)
    ):
        message = f"{label} must be one exact (Y, X) tuple of positive integers."
        raise ValueError(message)
    return value


@dataclass(frozen=True, slots=True)
class TrainScaling:
    """
    Persist the Train-only channel scaling of one run, the same way for both tasks.

    All tensors share one device and dtype. The serialized state always holds
    device-independent JSON-compatible CPU values, so its digest does not change
    when a runtime copy moves between supported devices.
    """

    task: str
    task_contract_digest: str
    data_contract_digest: str
    preprocessing: Mapping[str, Any]
    dataset_identity: Mapping[str, Any]
    train_membership_digest: str
    train_case_count: int
    spatial_shape: tuple[int, int]
    groups: Mapping[str, ChannelStatistics]
    horizon: float | None = None

    def __post_init__(self) -> None:
        """Isolate values and reject identity, group, shape, horizon or device drift."""
        task = domain.tasks.registry.get_task(self.task)
        if _require_sha256(self.task_contract_digest, label="task_contract_digest") != task.contract_digest:
            message = f"Scaling task contract does not match {self.task}."
            raise ValueError(message)
        if _require_sha256(self.data_contract_digest, label="data_contract_digest") != task.data_contract_digest:
            message = f"Scaling data contract does not match {self.task}."
            raise ValueError(message)
        preprocessing = _canonical_mapping(self.preprocessing, label="preprocessing")
        if preprocessing.get("kind") != PREPROCESSING_KIND or not isinstance(preprocessing.get("spatial_view"), dict):
            message = f"Scaling preprocessing must be one {PREPROCESSING_KIND!r} specification with a spatial view."
            raise ValueError(message)
        reference = STEADY_STATE_REFERENCE if self.task == STEADY_TASK else None
        if preprocessing.get("state_reference") != reference:
            message = (
                f"Scaling preprocessing of {self.task} must name the state reference {reference!r}, got "
                f"{preprocessing.get('state_reference')!r}; steady scalings saved before the nondimensionalization by the "
                "mean inlet pressure are not read. Start a new run."
            )
            raise ValueError(message)
        object.__setattr__(self, "preprocessing", MappingProxyType(preprocessing))
        object.__setattr__(self, "dataset_identity", MappingProxyType(_canonical_mapping(self.dataset_identity, label="dataset_identity")))
        _require_sha256(self.train_membership_digest, label="train_membership_digest")
        if isinstance(self.train_case_count, bool) or not isinstance(self.train_case_count, int) or self.train_case_count < 1:
            message = "train_case_count must be a positive integer."
            raise ValueError(message)
        shape = _positive_shape(self.spatial_shape, label="spatial_shape")
        view = preprocessing["spatial_view"]
        if (view.get("effective_ny"), view.get("effective_nx")) != shape:
            message = "Scaling spatial_shape does not match the effective grid of its preprocessing spatial view."
            raise ValueError(message)
        self._validate_horizon()
        self._validate_groups()

    def _validate_horizon(self) -> None:
        """Require the configured regular horizon exactly for the transient task."""
        if self.task != TRANSIENT_TASK:
            if self.horizon is not None:
                message = "Only the transient scaling carries a time horizon."
                raise ValueError(message)
            return
        if (
            isinstance(self.horizon, bool)
            or not isinstance(self.horizon, (int, float))
            or not math.isfinite(float(self.horizon))
            or float(self.horizon) <= 0.0
        ):
            message = "The transient scaling horizon must be one positive finite number."
            raise ValueError(message)
        object.__setattr__(self, "horizon", float(self.horizon))

    def _validate_groups(self) -> None:
        """Require exactly the task's channel groups on one device and dtype."""
        if not isinstance(self.groups, Mapping) or any(not isinstance(value, ChannelStatistics) for value in self.groups.values()):
            message = "Scaling groups must map names to ChannelStatistics."
            raise TypeError(message)
        expected = expected_groups(self.task)
        observed = {name: statistics.names for name, statistics in self.groups.items()}
        if observed != expected:
            message = f"Scaling groups {observed} do not match the {self.task} channel groups {expected}."
            raise ValueError(message)
        placements = {(statistics.mean.device, statistics.mean.dtype) for statistics in self.groups.values()}
        if len(placements) != 1:
            message = "All scaling groups must share device and dtype."
            raise ValueError(message)
        object.__setattr__(self, "groups", MappingProxyType({name: self.groups[name] for name in expected}))

    @property
    def device(self) -> torch.device:
        """Return the shared runtime tensor device."""
        return self.groups["state"].mean.device

    @property
    def dtype(self) -> torch.dtype:
        """Return the shared runtime tensor dtype."""
        return self.groups["state"].mean.dtype

    @property
    def state_scale(self) -> torch.Tensor:
        """Return the scale max(sigma, floor) of the encoded states (steady: dimensionless) that the data loss and decoding use."""
        return self.groups["state"].scale

    @property
    def physical_state(self) -> ChannelStatistics:
        """Return the Train statistics of the physical states, whose scale metric normalization and checkpoint selection divide by."""
        return self.groups["physical_state"] if self.task == STEADY_TASK else self.groups["state"]

    @property
    def regular_horizon(self) -> float:
        """Return the configured regular time horizon of the transient scaling."""
        if self.horizon is None:
            message = f"The {self.task} scaling carries no time horizon."
            raise ValueError(message)
        return self.horizon

    def group(self, name: str) -> ChannelStatistics:
        """Return one channel group of this task."""
        try:
            return self.groups[name]
        except KeyError as error:
            message = f"The {self.task} scaling has no {name!r} group."
            raise ValueError(message) from error

    @property
    def digest(self) -> str:
        """Return the device-independent SHA-256 of all persisted evidence."""
        encoded = json.dumps(self.state_dict(), ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def state_dict(self) -> dict[str, Any]:
        """Serialize exact identity and detached CPU statistics."""
        return {
            "schema_kind": SCALING_SCHEMA_KIND,
            "task": self.task,
            "task_contract_digest": self.task_contract_digest,
            "data_contract_digest": self.data_contract_digest,
            "preprocessing": _canonical_mapping(self.preprocessing, label="preprocessing"),
            "dataset_identity": _canonical_mapping(self.dataset_identity, label="dataset_identity"),
            "train_membership_digest": self.train_membership_digest,
            "scale_floor": SCALE_FLOOR,
            "train_case_count": self.train_case_count,
            "spatial_shape": list(self.spatial_shape),
            "horizon": self.horizon,
            "groups": {name: statistics.as_dict() for name, statistics in self.groups.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TrainScaling:
        """
        Strictly admit one JSON-compatible scaling state.

        Raises
        ------
        TypeError
            If the state or one of its groups is not a mapping.
        ValueError
            If the state is not the current ``train_channel_scaling`` schema (a neuralop normalizer or an earlier
            transient scaling included), or its keys, floor, groups or identity are inconsistent.

        """
        if not isinstance(state, Mapping):
            message = "Scaling state must be a mapping."
            raise TypeError(message)
        kind = state.get("schema_kind")
        if kind != SCALING_SCHEMA_KIND:
            message = (
                f"Scaling state {kind!r} is not the current {SCALING_SCHEMA_KIND!r} format; normalizers fitted before the "
                "common case-equal Train scaling of both tasks are not read. Start a new run."
            )
            raise ValueError(message)
        if set(state) != _STATE_KEYS:
            missing, unknown = sorted(_STATE_KEYS - set(state)), sorted(set(state) - _STATE_KEYS)
            message = f"Scaling state keys do not match the current schema; missing {missing}, unknown {unknown}."
            raise ValueError(message)
        if state["scale_floor"] != SCALE_FLOOR:
            message = f"Scaling state floor {state['scale_floor']!r} is not the current floor {SCALE_FLOOR}."
            raise ValueError(message)
        raw_groups = state["groups"]
        if not isinstance(raw_groups, Mapping) or any(not isinstance(value, Mapping) or set(value) != _GROUP_KEYS for value in raw_groups.values()):
            message = "Scaling state groups must map names to {'names', 'mean', 'std'}."
            raise TypeError(message)
        shape = state["spatial_shape"]
        return cls(
            task=state["task"],
            task_contract_digest=state["task_contract_digest"],
            data_contract_digest=state["data_contract_digest"],
            preprocessing=state["preprocessing"],
            dataset_identity=state["dataset_identity"],
            train_membership_digest=state["train_membership_digest"],
            train_case_count=state["train_case_count"],
            spatial_shape=tuple(shape) if isinstance(shape, list) else shape,
            horizon=state["horizon"],
            groups={
                name: ChannelStatistics(
                    names=tuple(value["names"]),
                    mean=torch.as_tensor(value["mean"], dtype=torch.float32),
                    std=torch.as_tensor(value["std"], dtype=torch.float32),
                )
                for name, value in raw_groups.items()
            },
        )

    def to(self, device: torch.device | str) -> TrainScaling:
        """Return an equivalent runtime scaling on one concrete device."""
        target = _concrete_device(device)
        return replace(self, groups={name: statistics.to(target) for name, statistics in self.groups.items()})

    def assess_spatial_compatibility(self, evaluation_shape: tuple[int, int]) -> dict[str, Any]:
        """Describe exact or broadcast-safe use on one evaluation grid."""
        shape = _positive_shape(evaluation_shape, label="evaluation_shape")
        decision = "SUPPORTED_EXACTLY" if shape == self.spatial_shape else "SUPPORTED_WITH_CONTRACT"
        return {
            "decision": decision,
            "training_shape": list(self.spatial_shape),
            "evaluation_shape": list(shape),
            "statistics_scope": "channelwise_pointwise_broadcastable",
            "shape_specific_values": [],
            "fitted_statistics_mutated": False,
            "reason": (
                "Evaluation shape equals the admitted Training shape."
                if decision == "SUPPORTED_EXACTLY"
                else "All admitted fitted statistics are channel vectors and broadcast pointwise without spatial interpolation."
            ),
        }

    def _value(self, value: Any, *, label: str, require_finite: bool = True) -> torch.Tensor:
        """Admit one runtime tensor on this scaling's device and dtype."""
        return _runtime_value(value, label=label, device=self.device, dtype=self.dtype, require_finite=require_finite)

    def state_reference(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Return the per-case reference P ``[B]`` of physical steady model inputs ``[B, C, Y, X]``.

        P is the mean prescribed inlet pressure; every steady encode and decode of the same cases takes it.

        Raises
        ------
        ValueError
            For the transient task, or if P of a case is not finite and positive (naming its batch position).

        """
        if self.task != STEADY_TASK:
            message = f"Only the steady representation has a per-case state reference, not {self.task}."
            raise ValueError(message)
        return _mean_inlet_pressure(self._value(inputs, label="physical model inputs"))

    def _per_case(self, reference: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor | None:
        """Return the steady reference ``[B]`` as a broadcast view ``[B, 1, ...]``; the transient encoding takes none."""
        if self.task != STEADY_TASK:
            if reference is not None:
                message = "The transient encoding takes no per-case reference; only the steady representation divides by one."
                raise ValueError(message)
            return None
        if reference is None:
            message = "The steady encoding requires the per-case reference P (state_reference of the physical inputs)."
            raise ValueError(message)
        tensor = self._value(reference, label="steady reference")
        if tensor.ndim != 1 or tensor.shape[0] != value.shape[0] or not bool((tensor > 0).all().item()):
            message = f"The steady reference must hold one positive value per case, shape ({value.shape[0]},), got {tuple(tensor.shape)}."
            raise ValueError(message)
        return tensor.view(-1, *([1] * (value.ndim - 1)))

    def encode_state(self, value: torch.Tensor, *, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Standardize physical state channels of a BCHW or BLCHW tensor; steady states are divided by ``reference`` first."""
        tensor = self._value(value, label="state")
        per_case = self._per_case(reference, tensor)
        if per_case is not None:
            tensor = tensor / per_case
        state = self.groups["state"]
        channels = len(state.names)
        return (tensor - _channel_view(state.mean, tensor, channels=channels)) / _channel_view(state.scale, tensor, channels=channels)

    def decode_state(self, value: torch.Tensor, *, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Return physical states from standardized ones, steady ones times ``reference``; non-finite outputs pass through."""
        tensor = self._value(value, label="standardized state", require_finite=False)
        per_case = self._per_case(reference, tensor)
        state = self.groups["state"]
        channels = len(state.names)
        decoded = tensor * _channel_view(state.scale, tensor, channels=channels) + _channel_view(state.mean, tensor, channels=channels)
        return decoded if per_case is None else decoded * per_case

    def encode_delta(self, value: torch.Tensor) -> torch.Tensor:
        """Scale increments with the state scale without centering, so exact zeros stay zero."""
        tensor = self._value(value, label="increment")
        return tensor / _channel_view(self.state_scale, tensor, channels=len(self.groups["state"].names))

    def decode_delta(self, value: torch.Tensor) -> torch.Tensor:
        """Decode scaled increments under the strict finite contract."""
        tensor = self._value(value, label="scaled increment")
        decoded = tensor * _channel_view(self.state_scale, tensor, channels=len(self.groups["state"].names))
        if not bool(torch.isfinite(decoded).all().item()):
            message = "Decoded increment is non-finite."
            raise FloatingPointError(message)
        return decoded

    def decode_delta_diagnostic(self, value: torch.Tensor) -> torch.Tensor:
        """Decode a model output while preserving IEEE non-finite diagnostic values."""
        tensor = self._value(value, label="diagnostic scaled increment", require_finite=False)
        return tensor * _channel_view(self.state_scale, tensor, channels=len(self.groups["state"].names))

    def encode_static(self, value: torch.Tensor) -> torch.Tensor:
        """Standardize static spatial channels of shape ``[B, C, Y, X]``."""
        tensor = self._value(value, label="static conditioning")
        static = self.groups["static"]
        if tensor.ndim != _SPATIAL_BATCH_RANK:
            message = f"Static conditioning must have shape [B,{len(static.names)},Y,X]."
            raise ValueError(message)
        channels = len(static.names)
        return (tensor - _channel_view(static.mean, tensor, channels=channels)) / _channel_view(static.scale, tensor, channels=channels)

    def encode_scalars(self, value: torch.Tensor) -> torch.Tensor:
        """Standardize scalar material-conditioning channels of shape ``[B, C]``."""
        tensor = self._value(value, label="scalar conditioning")
        scalar = self.group("scalar")
        if tensor.ndim != _VECTOR_BATCH_RANK or tensor.shape[1] != len(scalar.names):
            message = f"Scalar conditioning must have shape [B,{len(scalar.names)}]."
            raise ValueError(message)
        return (tensor - scalar.mean.view(1, -1)) / scalar.scale.view(1, -1)

    def encode_boundary(self, value: torch.Tensor) -> torch.Tensor:
        """
        Scale transient interval conditioning and neutralize absent startup evidence.

        Regular, ambient and startup temperatures share the Train state scale of ``T``.
        Regular and present startup humidity-ratio values share the fitted
        ``omega_boundary`` scale. The startup flag remains exactly binary.
        """
        tensor = self._value(value, label="boundary conditioning")
        omega = self.group("omega_boundary")
        horizon = self.regular_horizon
        if tensor.ndim != _VECTOR_BATCH_RANK or tensor.shape[1] != _BOUNDARY_CHANNELS:
            message = "Boundary conditioning must have shape [B,9]."
            raise ValueError(message)
        flag = tensor[:, 8]
        if not bool(((flag == 0) | (flag == 1)).all().item()):
            message = "startup_support_present must contain exact binary values."
            raise ValueError(message)
        state = self.groups["state"]
        temperature = state.names.index("T")
        temperature_mean, temperature_scale = state.mean[temperature], state.scale[temperature]
        omega_mean, omega_scale = omega.mean[0], omega.scale[0]
        encoded = tensor.clone()
        encoded[:, 0:2] = (encoded[:, 0:2] - temperature_mean) / temperature_scale
        encoded[:, 2:4] = (encoded[:, 2:4] - omega_mean) / omega_scale
        encoded[:, 4] = (encoded[:, 4] - temperature_mean) / temperature_scale
        encoded[:, 5] = encoded[:, 5] / horizon
        encoded[:, 6] = (encoded[:, 6] - temperature_mean) / temperature_scale
        encoded[:, 7] = (encoded[:, 7] - omega_mean) / omega_scale
        present = flag == 1
        encoded[:, 5:8] = torch.where(present[:, None], encoded[:, 5:8], torch.zeros_like(encoded[:, 5:8]))
        encoded[:, 8] = flag
        return encoded

    def input_affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the per-input mean and scale of the steady model inputs, boundary inputs (divided by P) taking their state's."""
        if self.task != STEADY_TASK:
            message = "Model-input encoding applies to steady_flow; the transient tensorizer assembles its inputs."
            raise ValueError(message)
        static, state = self.groups["static"], self.groups["state"]
        means: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for name in domain.tasks.registry.get_task(STEADY_TASK).input_names:
            source, channel = (state, STEADY_BOUNDARY_STATES[name]) if name in STEADY_BOUNDARY_STATES else (static, name)
            index = source.names.index(channel)
            means.append(source.mean[index])
            scales.append(source.scale[index])
        return torch.stack(means), torch.stack(scales)

    def _input_divisor(self, reference: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
        """Return the ``[B, C, 1, 1]`` divisor of steady model inputs: P for every boundary input, one for the others."""
        per_case = cast("torch.Tensor", self._per_case(reference, value))
        names = domain.tasks.registry.get_task(STEADY_TASK).input_names
        boundary = torch.tensor([name in STEADY_BOUNDARY_STATES for name in names], device=self.device).view(1, -1, 1, 1)
        return torch.where(boundary, per_case, torch.ones_like(per_case))

    def encode_inputs(self, value: torch.Tensor, *, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Standardize steady model inputs ``[B, C, Y, X]``; ``p_in_bc/reference`` uses the mean and scale of ``p/P``."""
        tensor = self._value(value, label="model inputs")
        mean, scale = self.input_affine()
        tensor = tensor / self._input_divisor(reference, tensor)
        return (tensor - _channel_view(mean, tensor, channels=mean.numel())) / _channel_view(scale, tensor, channels=scale.numel())

    def decode_inputs(self, value: torch.Tensor, *, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Return physical steady model inputs from standardized ones, the boundary inputs times ``reference``."""
        tensor = self._value(value, label="standardized model inputs")
        mean, scale = self.input_affine()
        decoded = tensor * _channel_view(scale, tensor, channels=scale.numel()) + _channel_view(mean, tensor, channels=mean.numel())
        return decoded * self._input_divisor(reference, tensor)


def preprocessing_spec(task: str, *, spatial_view: Mapping[str, int], tensorizer: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Return the preprocessing specification that the shared scaling identity binds.

    Parameters
    ----------
    task : str
        Task identifier.
    spatial_view : Mapping[str, int]
        Canonical and effective grid evidence of the configured ``data.spatial_stride``.
    tensorizer : Mapping[str, Any] | None, optional
        The transient tensorizer identity; required for the transient task and refused for steady flow.

    """
    expected_groups(task)
    if (task == TRANSIENT_TASK) != (tensorizer is not None):
        message = "A tensorizer identity belongs to the transient preprocessing specification only."
        raise ValueError(message)
    spec: dict[str, Any] = {"kind": PREPROCESSING_KIND, "spatial_view": dict(spatial_view)}
    if task == STEADY_TASK:
        spec["state_reference"] = STEADY_STATE_REFERENCE
    if tensorizer is not None:
        spec["tensorizer"] = dict(tensorizer)
    return spec


def admit_scaling_state(state: Mapping[str, Any], *, identity: Mapping[str, Any]) -> TrainScaling:
    """
    Rebuild one saved scaling state and require it to repeat the expected envelope identity.

    Parameters
    ----------
    state : Mapping[str, Any]
        Scaling state of an identity-admitted envelope.
    identity : Mapping[str, Any]
        Expected scaling identity (``datasets.preprocessing.scaling.scaling_identity``).

    Returns
    -------
    TrainScaling
        The strictly rebuilt scaling.

    Raises
    ------
    ValueError
        If the state is not the current format, or its task, Dataset identity, Train membership or preprocessing differs
        from the identity, naming the field.

    """
    scaling = TrainScaling.from_state_dict(state)
    expected = {
        "task": identity["task"],
        "task_contract_digest": identity["task_contract_digest"],
        "dataset_identity": _canonical_mapping(identity["dataset"], label="dataset_identity"),
        "train_membership_digest": identity["train_membership_digest"],
        "preprocessing": _canonical_mapping(identity["preprocessing"], label="preprocessing"),
    }
    observed = {
        "task": scaling.task,
        "task_contract_digest": scaling.task_contract_digest,
        "dataset_identity": dict(scaling.dataset_identity),
        "train_membership_digest": scaling.train_membership_digest,
        "preprocessing": dict(scaling.preprocessing),
    }
    for field, value in expected.items():
        if observed[field] != value:
            message = f"Scaling state {field} does not match the expected scaling identity."
            raise ValueError(message)
    return scaling


def build_scaling(
    task: str,
    *,
    preprocessing: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    spatial_shape: tuple[int, int],
    moments: Mapping[str, CaseWeightedMoments],
    horizon: float | None = None,
) -> TrainScaling:
    """Assemble one scaling from the finished case-weighted moments of every group of the task."""
    names = expected_groups(task)
    counts = {moment.case_count for moment in moments.values()}
    if set(moments) != set(names) or len(counts) != 1:
        message = f"Scaling fit must close the same cases in every group of {task}: {sorted(names)}."
        raise ValueError(message)
    groups = {}
    for name, channels in names.items():
        mean, std = moments[name].statistics()
        groups[name] = ChannelStatistics(names=channels, mean=mean, std=std)
    spec = domain.tasks.registry.get_task(task)
    return TrainScaling(
        task=task,
        task_contract_digest=spec.contract_digest,
        data_contract_digest=spec.data_contract_digest,
        preprocessing=preprocessing,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        train_case_count=counts.pop(),
        spatial_shape=spatial_shape,
        groups=groups,
        horizon=horizon,
    )


def fit_steady_scaling(
    items: Iterable[Mapping[str, Any]],
    *,
    preprocessing: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
) -> TrainScaling:
    """
    Fit the steady scaling from Train items, each one complete case, without concatenating them.

    Every item holds inputs ``x`` ``[C_in, Y, X]`` and outputs ``y`` ``[C_out, Y, X]`` of one case, so
    every case weighs one. The ``state`` moments are fitted on ``y/P`` with P the case's mean prescribed
    inlet pressure, the ``physical_state`` moments on ``y``; ``p_in_bc`` is not fitted, ``p_in_bc/P`` is
    encoded with the dimensionless state scale of ``p``.
    """
    task = domain.tasks.registry.get_task(STEADY_TASK)
    names = expected_groups(STEADY_TASK)
    static_indices = [task.input_names.index(name) for name in names["static"]]
    moments = {name: CaseWeightedMoments(len(channels)) for name, channels in names.items()}
    spatial_shape: tuple[int, int] | None = None
    for position, item in enumerate(items):
        if not isinstance(item, Mapping):
            message = "Steady scaling items must be mappings."
            raise TypeError(message)
        inputs, outputs = train_item_tensor(item, "x"), train_item_tensor(item, "y")
        if inputs.ndim != _UNBATCHED_SPATIAL_RANK or outputs.ndim != _UNBATCHED_SPATIAL_RANK:
            message = "Steady scaling items must be unbatched [C, Y, X] tensors."
            raise ValueError(message)
        shape = (int(inputs.shape[1]), int(inputs.shape[2]))
        if inputs.shape[0] != task.in_channels or tuple(outputs.shape) != (task.out_channels, *shape) or spatial_shape not in {None, shape}:
            message = "Steady scaling items must share one grid and carry the task's input and output channels."
            raise ValueError(message)
        spatial_shape = shape
        reference = _mean_inlet_pressure(inputs[None], offset=position)[0]
        moments["static"].update(inputs[static_indices])
        moments["state"].update(outputs / reference)
        moments["physical_state"].update(outputs)
        for moment in moments.values():
            moment.close_case()
    if spatial_shape is None:
        message = "Steady scaling fit requires at least one Train case."
        raise ValueError(message)
    return build_scaling(
        STEADY_TASK,
        preprocessing=preprocessing,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        spatial_shape=spatial_shape,
        moments=moments,
    )
