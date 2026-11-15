"""Fit the transient Train scaling from ordered one-step Train items, one case at a time.

Responsibilities:
  - Admit Train items only in canonical grouped case and consecutive hourly order, requiring each transition to start
    at the stored state its predecessor ended at
  - Collapse an adjacent duplicate transition only when its stored start and end states repeat exactly
  - Accumulate case-equal statistics: states and the derived grain moisture per state, the inlet humidity ratio per
    transition, static fields and material scalars once per case
  - Build the common Train scaling of both tasks from those statistics

Design principles:
  - Statistics consume stored Train states only; no state is reconstructed from a float32 increment
  - Every Train case weighs one; its states, the stored terminal state included, weigh equally within it
  - Static fields and material scalars must repeat exactly within a case, so counting them once per case loses nothing

This module does NOT:
  - Load HDF5 or select Dataset memberships
  - Define, persist or apply the scaling state; ``datasets.preprocessing.normalization`` owns it for both tasks

Notes:
  Fit counting:
    - A case with n unique transitions contributes n + 1 stored states (every transition's start state and the last
      transition's ``next_state``), each with weight 1 / (n + 1) inside the case; ``w_gr`` uses the ``f_surf`` of the case.
    - The inlet humidity ratio contributes both regular endpoints of every unique transition and a present startup value.

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Final, cast

import torch

from src import datasets, domain

_TASK: Final = "transient_drying"
_STATE_CHANNELS: Final = 4
_STATIC_CHANNELS: Final = 7
_BOUNDARY_CHANNELS: Final = 9
_SCALAR_CHANNELS: Final = 8
_UNBATCHED_SPATIAL_RANK: Final = 3
_TIME_TOLERANCE: Final = 1.0e-6
_ITEM_TENSORS: Final = ("state", "next_state", "static", "boundary", "scalars")


def _grain_moisture(state: torch.Tensor, f_surf: float) -> torch.Tensor:
    """Return the derived float64 grain moisture of one [4,Y,X] state as one channel."""
    names = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.dynamic_state)
    surface, internal = state[names.index("w_surf")], state[names.index("w_int")]
    return torch.from_numpy(domain.moisture.granular_water_content(surface.numpy(), internal.numpy(), f_surf)).unsqueeze(0)


class _CaseOrderedEvidence:
    """Validate canonical case chains while accumulating case-equal moments of every scaling group."""

    def __init__(self) -> None:
        """Initialize empty ordered Train evidence."""
        groups = datasets.preprocessing.normalization.expected_groups(_TASK)
        self.moments = {name: datasets.preprocessing.normalization.CaseWeightedMoments(len(channels)) for name, channels in groups.items()}
        self._seen_simulations: set[str] = set()
        self._active_simulation: str | None = None
        self._case_static: torch.Tensor | None = None
        self._case_scalars: torch.Tensor | None = None
        self._previous_state: torch.Tensor | None = None
        self._previous_next_state: torch.Tensor | None = None
        self._previous_f_surf: float | None = None
        self._previous_index_n: int | None = None
        self._previous_index_next: int | None = None
        self._previous_t_next: float | None = None

    def _finish_active_case(self) -> None:
        """Admit the stored terminal state and close the active case in every group."""
        if self._active_simulation is None:
            return
        if self._previous_next_state is None or self._previous_f_surf is None:
            message = "Transient scaling case closed without a transition."
            raise RuntimeError(message)
        terminal = self._previous_next_state
        self.moments["state"].update(terminal)
        self.moments["grain_moisture"].update(_grain_moisture(terminal, self._previous_f_surf))
        for moment in self.moments.values():
            moment.close_case()
        self._active_simulation = None
        self._previous_state = None
        self._previous_next_state = None

    def _begin_case(self, simulation: str, *, index_n: int, t_n: float, static: torch.Tensor, scalars: torch.Tensor) -> None:
        """Finish the prior case, admit one canonical zero-time case start and count its static evidence once."""
        if simulation in self._seen_simulations:
            message = "Transient scaling items must retain canonical grouped case order."
            raise ValueError(message)
        self._finish_active_case()
        if index_n != 0 or not math.isclose(t_n, 0.0, rel_tol=0.0, abs_tol=_TIME_TOLERANCE):
            message = "Each transient scaling case must begin at its canonical zero-time transition."
            raise ValueError(message)
        self._seen_simulations.add(simulation)
        self._active_simulation = simulation
        self._previous_index_n = None
        self._previous_index_next = None
        self._previous_t_next = None
        self.moments["static"].update(static)
        self.moments["scalar"].update(scalars)
        self._case_static = static
        self._case_scalars = scalars

    def _admit_duplicate(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
        """Require one adjacent duplicate to repeat its stored start and end states exactly."""
        if self._previous_state is None or self._previous_next_state is None:
            message = "Transient scaling duplicate transition evidence is incomplete."
            raise RuntimeError(message)
        if not torch.equal(self._previous_state, state):
            message = "Duplicate direct transient state evidence disagrees exactly."
            raise ValueError(message)
        if not torch.equal(self._previous_next_state, next_state):
            message = "Duplicate transient transition evidence disagrees exactly."
            raise ValueError(message)

    def _require_chain(self, state: torch.Tensor, *, index_n: int, t_n: float) -> None:
        """Require one new transition to start exactly at the stored state the previous one ended at."""
        if self._previous_index_next is None:
            return
        if (
            self._previous_t_next is None
            or index_n != self._previous_index_next
            or not math.isclose(t_n, self._previous_t_next, rel_tol=0.0, abs_tol=_TIME_TOLERANCE)
        ):
            message = "Transient scaling items must retain canonical consecutive case/time order."
            raise ValueError(message)
        if self._previous_next_state is None:
            message = "Transient scaling transition chain evidence is incomplete."
            raise RuntimeError(message)
        if not torch.equal(state, self._previous_next_state):
            message = "Direct transient state evidence disagrees with the stored state its predecessor ended at."
            raise ValueError(message)

    def admit(
        self,
        item: Mapping[str, torch.Tensor],
        *,
        f_surf: float,
        simulation: str,
        index_n: int,
        index_next: int,
        t_n: float,
        t_next: float,
    ) -> None:
        """Admit one ordered raw transition and retain at most one stored endpoint pair."""
        state, next_state, static, scalars, boundary = (item[name] for name in ("state", "next_state", "static", "scalars", "boundary"))
        if simulation != self._active_simulation:
            self._begin_case(simulation, index_n=index_n, t_n=t_n, static=static, scalars=scalars)
        elif (
            self._case_static is None
            or self._case_scalars is None
            or not torch.equal(static, self._case_static)
            or not torch.equal(scalars, self._case_scalars)
        ):
            message = "Static fields and material scalars must repeat exactly within one transient Train case."
            raise ValueError(message)
        if self._previous_index_n == index_n and self._previous_index_next == index_next:
            self._admit_duplicate(state, next_state)
            return
        self._require_chain(state, index_n=index_n, t_n=t_n)
        self.moments["state"].update(state)
        self.moments["grain_moisture"].update(_grain_moisture(state, f_surf))
        self.moments["omega_boundary"].update(boundary[2:4].reshape(1, -1))
        if float(boundary[8].item()) == 1.0:
            self.moments["omega_boundary"].update(boundary[7:8].reshape(1, -1))
        self._previous_state = state
        self._previous_next_state = next_state
        self._previous_f_surf = f_surf
        self._previous_index_n = index_n
        self._previous_index_next = index_next
        self._previous_t_next = t_next

    def finish(self) -> dict[str, datasets.preprocessing.normalization.CaseWeightedMoments]:
        """Close the last case and return the case-equal moments of every group."""
        if not self._seen_simulations:
            message = "Transient scaling fit requires at least one Train item."
            raise ValueError(message)
        self._finish_active_case()
        return dict(self.moments)


def _scalar_time(time: Mapping[str, Any], name: str) -> float:
    """Return one finite scalar temporal value."""
    value = time.get(name)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
        message = f"time.{name} must contain one finite scalar value."
        raise ValueError(message)
    return float(value.detach().cpu().item())


def _metadata_identity(metadata: Mapping[str, Any]) -> tuple[str, int, int]:
    """Return strict simulation and transition-index identity."""
    simulation = metadata.get("simulation_case_id")
    index_n = metadata.get("time_index_n")
    index_next = metadata.get("time_index_n_plus_1")
    if not isinstance(simulation, str) or not simulation:
        message = "simulation_case_id must be one non-empty string."
        raise ValueError(message)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (index_n, index_next)):
        message = "time_index_n and time_index_n_plus_1 must be non-negative integers."
        raise ValueError(message)
    index_n, index_next = cast("int", index_n), cast("int", index_next)
    if index_next <= index_n:
        message = "Transient time indexes must advance."
        raise ValueError(message)
    return simulation, index_n, index_next


def _validate_fit_shapes(item: Mapping[str, torch.Tensor], *, expected_spatial_shape: tuple[int, int] | None) -> tuple[int, int]:
    """Require exact unbatched one-step Train item shapes."""
    state = item["state"]
    if state.ndim != _UNBATCHED_SPATIAL_RANK or state.shape[0] != _STATE_CHANNELS:
        message = "Train state must have shape [4,Y,X]."
        raise ValueError(message)
    spatial_shape = (int(state.shape[1]), int(state.shape[2]))
    if min(spatial_shape) < 1:
        message = "Train state spatial axes must be non-empty."
        raise ValueError(message)
    expected = {
        "next_state": (_STATE_CHANNELS, *spatial_shape),
        "static": (_STATIC_CHANNELS, *spatial_shape),
        "boundary": (_BOUNDARY_CHANNELS,),
        "scalars": (_SCALAR_CHANNELS,),
    }
    for label, shape in expected.items():
        if tuple(item[label].shape) != shape:
            message = f"Train {label} must have shape {shape}, got {tuple(item[label].shape)}."
            raise ValueError(message)
    if expected_spatial_shape is not None and spatial_shape != expected_spatial_shape:
        message = "Train items do not share one spatial shape."
        raise ValueError(message)
    if float(item["boundary"][8].item()) not in {0.0, 1.0}:
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
    preprocessing: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
    train_membership_digest: str,
    horizon: float,
) -> datasets.preprocessing.normalization.TrainScaling:
    """
    Fit the case-equal transient scaling from canonical one-step Train items only.

    Items must retain the package index's grouped, consecutive case/time order and carry the stored
    ``next_state`` their transition ends at. Each transition must start exactly at the stored state its
    predecessor ended at, and the last transition's stored ``next_state`` is a case's terminal state, so
    no state is reconstructed from a rounded float32 increment. Stable float64 moments avoid retaining
    source tensors; Validation/OOD data cannot influence the fitted statistics.
    """
    resolved_horizon = _positive_horizon(horizon)
    evidence = _CaseOrderedEvidence()
    spatial_shape: tuple[int, int] | None = None
    f_surf_index = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.scalar_conditioning).index("f_surf")
    for raw in items:
        if not isinstance(raw, Mapping):
            message = "Transient scaling items must be mappings."
            raise TypeError(message)
        item = {name: datasets.preprocessing.normalization.train_item_tensor(raw, name) for name in _ITEM_TENSORS}
        spatial_shape = _validate_fit_shapes(item, expected_spatial_shape=spatial_shape)
        metadata, time = raw.get("metadata"), raw.get("time")
        if not isinstance(metadata, Mapping) or not isinstance(time, Mapping):
            message = "Train items require metadata and time mappings."
            raise TypeError(message)
        simulation, index_n, index_next = _metadata_identity(metadata)
        t_n, t_next, dt = (_scalar_time(time, name) for name in ("t_n", "t_n_plus_1", "dt"))
        regular = math.isclose(dt, 1.0, rel_tol=0.0, abs_tol=_TIME_TOLERANCE)
        if not regular or not math.isclose(t_next - t_n, dt, rel_tol=0.0, abs_tol=_TIME_TOLERANCE):
            message = "Transient scaling requires fixed regular 1h transitions."
            raise ValueError(message)
        if t_n < 0.0 or t_next > resolved_horizon + _TIME_TOLERANCE:
            message = "Train time evidence lies outside the configured horizon."
            raise ValueError(message)
        evidence.admit(
            item,
            f_surf=float(item["scalars"][f_surf_index].item()),
            simulation=simulation,
            index_n=index_n,
            index_next=index_next,
            t_n=t_n,
            t_next=t_next,
        )
    if spatial_shape is None:
        message = "Transient scaling fit requires at least one Train item."
        raise ValueError(message)
    return datasets.preprocessing.normalization.build_scaling(
        _TASK,
        preprocessing=preprocessing,
        dataset_identity=dataset_identity,
        train_membership_digest=train_membership_digest,
        spatial_shape=spatial_shape,
        moments=evidence.finish(),
        horizon=resolved_horizon,
    )
