"""Measure the paired field-test endpoints of two cases of one task in physical units.

Responsibilities:
  - Read the fields, stored times, per-hour wet fractions and completion evidence of one Source-format ``case.h5``
    through the Generation field-test output reader
  - Own the field-test regions: 16 x 16 cell patches of the interior, the wall strips and the inlet and outlet rows
  - Measure steady relative L2, patch maxima and wall-minus-core contrast of p, u, v and speed, and the same input norms
  - Measure transient per-hour RMSE and patch maxima, increment differences over the shared stored window, outlet edge
    means and completion, plus the wet-fraction change at the window's end
  - Measure the envelope a tolerance table or a replicate needs: every endpoint plus the increments at every window end
  - Name every endpoint once and hash the endpoint code, so tolerances and verdicts share one definition

Design principles:
  - One function measures a pair, whether the second case is a field-test variant or a model prediction
  - States are compared only at stored hours both cases hold; the shared stored window is the run of consecutive stored
    hours both hold from their first common hour, and its increments are named by its end h, so a tolerance is always
    looked up over the same window
  - A censored duration is a lower bound, never a value; an endpoint without support is missing (``None``)
  - Input-field norms, initial-hour states and the horizon wet fraction are reported but never enter a verdict

This module does NOT:
  - Read rosters, seals or trained runs, or decide verdicts
  - Validate a ``case.h5`` beyond the members it reads; the Generation writer owns the format and the reached rule
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import h5py
import numpy as np
import torch

from src import common, domain, generation, learning

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

PATCH_CELLS: Final = 16
STEADY_OUTPUTS: Final = ("p", "u", "v", "speed")
STEADY_INPUTS: Final = ("Kxx", "Kxy", "Kyy", "eps_bed", "p_in_bc")
TRANSIENT_STATES: Final = ("T", "phi", "w_surf", "w_int")
OUTLET_STATES: Final = ("T", "phi")
DURATION: Final = "transient/duration"
WET_FRACTION_HORIZON: Final = "transient/wet_fraction_horizon"
HOUR_ATOL: Final = 1.0e-6
_HOUR_PREFIX: Final = "@h="
ENDPOINT_PROJECTION: Final = common.code_identity.CodeProjection(
    name="field_test_endpoints",
    files=("analysis/field_tests/field_tests_endpoints.py", "generation/field_tests/generation_field_tests_outputs.py"),
)
_STEADY_TASK: Final = domain.tasks.registry.get_task("steady_flow")


@dataclass(frozen=True, slots=True)
class Completion:
    """Stop time, target flag, final dry-mass fraction above target, horizon and target definition of one transient case."""

    t_stop: float
    reached: bool
    wet_fraction_final: float
    horizon: float
    fraction_limit: float
    target_wet_basis: float


@dataclass(frozen=True, slots=True)
class CaseFields:
    """
    Hold the compared fields of one case on its ``[Y, X]`` grid, y-min the inlet row and x-min/x-max the walls.

    Attributes
    ----------
    task : str
        ``"steady_flow"`` or ``"transient_drying"``.
    fields : Mapping[str, numpy.ndarray]
        Steady outputs and inputs by name, ``[Y, X]`` in physical units; empty for a transient case.
    times : numpy.ndarray | None
        Stored hours ``[N]`` of a transient case.
    states : numpy.ndarray | None
        Stored states ``[N, 4, Y, X]`` in ``TRANSIENT_STATES`` order.
    completion : Completion | None
        Completion evidence of a transient case.
    wet_fractions : numpy.ndarray | None
        Dry-mass fraction above target ``[N]`` at each stored hour, from the case's global diagnostics; ``None`` for a
        prediction.

    """

    task: str
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)
    times: np.ndarray | None = None
    states: np.ndarray | None = None
    completion: Completion | None = None
    wet_fractions: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One endpoint value with its contributor count; ``censored`` marks a duration lower bound."""

    value: float | None
    contributors: int
    verdict: bool = True
    censored: bool = False


def endpoint_code_sha256() -> str:
    """Return the syntax digest of the endpoint definitions recorded by tolerances and checked by verdicts."""
    return ENDPOINT_PROJECTION.sha256()


def read_case_fields(path: Path | str) -> CaseFields:
    """
    Read one Source-format ``case.h5`` into the compared fields of its task.

    Parameters
    ----------
    path : Path | str
        A Source ``case.h5`` or a field-test member's copy of one.

    Returns
    -------
    CaseFields
        Steady outputs, speed and inputs, or transient stored states with per-hour wet fractions and exact completion
        evidence.

    Raises
    ------
    ValueError
        If the file is not a Generation source case of a supported profile, or a transient case lacks its final status.

    """
    outputs = generation.field_tests.outputs
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_kind") != "generation_source_case_hdf5":
            msg = f"{path} is not a Generation source case."
            raise ValueError(msg)
        task = str(handle.attrs["simulation_profile"])
        if task == "steady_flow":
            static = outputs.named_values(handle, "static/fields")
            values = {name: static[name] for name in ("p", "u", "v", *STEADY_INPUTS)}
            values["speed"] = np.hypot(values["u"], values["v"])
            return CaseFields(task=task, fields=values)
        if task != "transient_drying":
            msg = f"Unsupported field-test profile {task!r} in {path}."
            raise ValueError(msg)
        transient = outputs.dataset(handle, "transient/fields")
        if tuple(json.loads(str(transient.attrs["field_names"]))) != TRANSIENT_STATES:
            msg = f"{path} does not store the transient states {list(TRANSIENT_STATES)} in this order."
            raise ValueError(msg)
        final = outputs.completion(handle)
        if final is None:
            msg = f"{path} lacks the transient final status."
            raise ValueError(msg)
        times = np.asarray(outputs.dataset(handle, "time")[()], dtype=np.float64)
        completion = Completion(
            t_stop=final.t_final,
            reached=final.reached,
            wet_fraction_final=final.wet_fraction_final,
            horizon=final.horizon,
            fraction_limit=final.fraction_limit,
            target_wet_basis=float(outputs.named_values(handle, "scalar/values")["X_target_wb"]),
        )
        # The global diagnostics hold the stored hours first, then the exact stop.
        wet_fractions = outputs.named_values(handle, "global/values", axis=-1)["f_wet_dm"][: times.size]
        return CaseFields(
            task=task, times=times, states=np.asarray(transient[()], dtype=np.float32), completion=completion, wet_fractions=wet_fractions
        )


def _tile_starts(size: int) -> list[int]:
    """Return patch starts covering one axis with full patches, the last one flush with the far edge."""
    if size < PATCH_CELLS:
        msg = f"Field-test patches need at least {PATCH_CELLS} cells per axis, got {size}."
        raise ValueError(msg)
    starts = list(range(0, size - PATCH_CELLS + 1, PATCH_CELLS))
    if starts[-1] != size - PATCH_CELLS:
        starts.append(size - PATCH_CELLS)
    return starts


def patches(shape: tuple[int, int], region: str) -> Iterator[tuple[slice, slice]]:
    """
    Yield the ``(rows, columns)`` slices of the patches of one region.

    ``wall`` takes the patches of the first and last column strip, ``inlet`` and ``outlet`` those of the first and last
    row strip, and ``interior`` every other patch.
    """
    rows, columns = _tile_starts(shape[0]), _tile_starts(shape[1])
    for row in rows:
        for column in columns:
            on_wall = column in {columns[0], columns[-1]}
            on_inlet, on_outlet = row == rows[0], row == rows[-1]
            selected = {
                "wall": on_wall,
                "inlet": on_inlet,
                "outlet": on_outlet,
                "interior": not (on_wall or on_inlet or on_outlet),
            }
            if region not in selected:
                msg = f"Unknown field-test region {region!r}."
                raise ValueError(msg)
            if selected[region]:
                yield slice(row, row + PATCH_CELLS), slice(column, column + PATCH_CELLS)


def _patch_max(difference: np.ndarray, region: str, *, verdict: bool = True) -> Endpoint:
    """Return the largest patch-mean absolute difference of one region and its patch count."""
    shape = (int(difference.shape[0]), int(difference.shape[1]))
    means = [float(np.mean(np.abs(difference[rows, columns]))) for rows, columns in patches(shape, region)]
    return Endpoint(max(means) if means else None, len(means), verdict=verdict)


def _relative_l2(reference: np.ndarray, other: np.ndarray, *, floor: float) -> Endpoint:
    """Return the shared case metric's relative L2 of ``other`` against ``reference``, ``None`` below the floor."""
    sums = learning.metrics.metrics.field_statistics(torch.from_numpy(other[None, None]), torch.from_numpy(reference[None, None]))[0, 0]
    return Endpoint(learning.metrics.metrics.case_relative_norm(sums, norm="l2", floor=floor), reference.size)


def _wall_core_contrast(field_values: np.ndarray) -> float:
    """Return the mean over both wall strips minus the mean over the remaining core columns."""
    wall = np.zeros(field_values.shape[1], dtype=bool)
    wall[:PATCH_CELLS] = wall[-PATCH_CELLS:] = True
    return float(field_values[:, wall].mean() - field_values[:, ~wall].mean())


def _map_endpoints(prefix: str, a: np.ndarray, b: np.ndarray, relative: Endpoint, *, verdict: bool) -> dict[str, Endpoint]:
    """Return one map's relative L2, interior, wall and inlet patch maxima and the change of its wall-minus-core contrast."""
    result = {f"{prefix}/relative_l2": relative}
    for region in ("interior", "wall", "inlet"):
        result[f"{prefix}/patch_{region}"] = _patch_max(b - a, region, verdict=verdict)
    result[f"{prefix}/wall_core_contrast"] = Endpoint(abs(_wall_core_contrast(b) - _wall_core_contrast(a)), a.size, verdict=verdict)
    return result


def steady_endpoints(reference: CaseFields, other: CaseFields) -> dict[str, Endpoint]:
    """
    Measure every steady endpoint of ``other`` against ``reference``.

    Outputs p, u, v and speed get relative L2 (with the task's reference floors; speed uses the velocity floor), the
    largest interior, wall and inlet patch-mean absolute difference, and the change of the wall-minus-core contrast.
    The input maps get the same measures, reported only; a prediction without inputs has none.
    """
    result: dict[str, Endpoint] = {}
    for name in STEADY_OUTPUTS:
        a, b = reference.fields[name], other.fields[name]
        floor = _STEADY_TASK.reference_floor("u" if name == "speed" else name)
        result.update(_map_endpoints(f"steady/{name}", a, b, _relative_l2(a, b, floor=floor), verdict=True))
    for name in STEADY_INPUTS:
        if name in reference.fields and name in other.fields:
            a, b = reference.fields[name], other.fields[name]
            norm = float(np.linalg.norm(a))
            relative = Endpoint(float(np.linalg.norm(b - a)) / norm if norm > 0.0 else None, a.size, verdict=False)
            result.update(_map_endpoints(f"input/{name}", a, b, relative, verdict=False))
    return result


def common_hours(reference: CaseFields, other: CaseFields) -> list[tuple[int, int]]:
    """Return the index pairs of the stored hours both cases hold, in time order."""
    if reference.times is None or other.times is None:
        msg = "Transient endpoints need stored times on both cases."
        raise ValueError(msg)
    pairs: list[tuple[int, int]] = []
    for index, hour in enumerate(reference.times):
        matches = np.flatnonzero(np.abs(other.times - hour) <= HOUR_ATOL)
        if matches.size:
            pairs.append((index, int(matches[0])))
    return pairs


def hour_label(hour: float) -> str:
    """Return the endpoint suffix of one stored hour."""
    return f"{_HOUR_PREFIX}{hour:g}"


def split_hour(name: str) -> tuple[str, float | None]:
    """Return an endpoint name's quantity (channel and measure) and its stored hour or window end, None without one."""
    quantity, prefix, hour = name.partition(_HOUR_PREFIX)
    return quantity, float(hour) if prefix else None


def duration_endpoint(reference: Completion, other: Completion) -> Endpoint:
    """
    Return the duration difference, or the censoring lower bound when exactly one case missed the target.

    Both reached gives ``|t_stop(other) - t_stop(reference)|``. One censored gives ``horizon - t_stop`` of the case that
    reached, flagged ``censored``. Both censored gives no value: their wet fractions bound nothing.
    """
    if reference.reached and other.reached:
        return Endpoint(abs(other.t_stop - reference.t_stop), 1)
    if reference.reached != other.reached:
        reached = reference if reference.reached else other
        return Endpoint(max(reached.horizon - reached.t_stop, 0.0), 1, censored=True)
    return Endpoint(None, 0)


def _trajectories(reference: CaseFields, other: CaseFields) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reference's stored hours and both stored state trajectories, refusing a case without them."""
    if reference.states is None or other.states is None or reference.times is None:
        msg = "Transient endpoints need stored states on both cases."
        raise ValueError(msg)
    return reference.times, reference.states, other.states


def _shared_window(reference: CaseFields, other: CaseFields) -> list[tuple[int, int]]:
    """Return the index pairs of the shared stored window: the consecutive stored hours both cases hold from their first common hour."""
    window: list[tuple[int, int]] = []
    for pair in common_hours(reference, other):
        if window and (pair[0] != window[-1][0] + 1 or pair[1] != window[-1][1] + 1):
            break
        window.append(pair)
    return window


def _increment_squares(reference: CaseFields, other: CaseFields, window: Sequence[tuple[int, int]]) -> np.ndarray:
    """Return the spatial means of the squared increment differences over the intervals of one window, ``[interval, state]``."""
    _times, ref_states, other_states = _trajectories(reference, other)
    squares = [
        np.mean(
            (
                (other_states[b1].astype(np.float64) - other_states[b0].astype(np.float64))
                - (ref_states[a1].astype(np.float64) - ref_states[a0].astype(np.float64))
            )
            ** 2,
            axis=(-2, -1),
        )
        for (a0, b0), (a1, b1) in pairwise(window)
    ]
    return np.asarray(squares, dtype=np.float64).reshape(max(len(window) - 1, 0), len(TRANSIENT_STATES))


def _window(squares: np.ndarray, end: float, cells: int) -> dict[str, Endpoint]:
    """Return the peak increment difference and the trajectory RMS of every state over one stored window ending at ``end``."""
    result: dict[str, Endpoint] = {}
    for channel, name in enumerate(TRANSIENT_STATES):
        column = squares[:, channel]
        contributors = int(column.size) * cells
        result[f"transient/{name}/increment_peak{hour_label(end)}"] = Endpoint(math.sqrt(float(column.max())), contributors)
        result[f"transient/{name}/increment_rms{hour_label(end)}"] = Endpoint(math.sqrt(float(column.mean())), contributors)
    return result


def _horizon_wet_fraction(reference: CaseFields, other: CaseFields, window: Sequence[tuple[int, int]]) -> Endpoint:
    """Return the wet-fraction change of both cases' states at the end of the shared stored window; unavailable without a state there."""
    if not window or reference.wet_fractions is None or other.wet_fractions is None:
        return Endpoint(None, 0, verdict=False)
    ref_index, other_index = window[-1]
    return Endpoint(abs(float(other.wet_fractions[other_index]) - float(reference.wet_fractions[ref_index])), 1, verdict=False)


def transient_endpoints(reference: CaseFields, other: CaseFields) -> dict[str, Endpoint]:
    """
    Measure every transient endpoint of ``other`` against ``reference`` at their common stored hours.

    Per state and common hour: spatial RMSE and the largest wall, inlet and outlet patch-mean absolute difference, and per
    outlet state the change of the outlet edge mean; the initial hour is reported only, because it is a rollout input
    without a yardstick. Over the shared stored window, ending at the common horizon h: the startup increment difference
    (RMS over space of the first interval), the peak increment difference (largest interval RMS) and the trajectory RMS,
    both named by h. Completion: duration (or its censoring bound), and the wet-fraction change at h, reported only.
    """
    times, ref_states, other_states = _trajectories(reference, other)
    window = _shared_window(reference, other)
    squares = _increment_squares(reference, other, window)
    startup = len(window) > 1 and window[0][0] == 0
    result: dict[str, Endpoint] = {}
    cells = int(np.prod(ref_states.shape[-2:]))
    for ref_index, other_index in common_hours(reference, other):
        verdict = ref_index != 0
        suffix = hour_label(float(times[ref_index]))
        difference = other_states[other_index].astype(np.float64) - ref_states[ref_index].astype(np.float64)
        for channel, name in enumerate(TRANSIENT_STATES):
            result[f"transient/{name}/rmse{suffix}"] = Endpoint(float(np.sqrt(np.mean(difference[channel] ** 2))), cells, verdict=verdict)
            for region in ("wall", "inlet", "outlet"):
                result[f"transient/{name}/patch_{region}{suffix}"] = _patch_max(difference[channel], region, verdict=verdict)
        for name in OUTLET_STATES:
            channel = TRANSIENT_STATES.index(name)
            result[f"transient/outlet_{name}{suffix}"] = Endpoint(float(abs(difference[channel, -1].mean())), difference.shape[-1], verdict=verdict)
    for channel, name in enumerate(TRANSIENT_STATES):
        value = math.sqrt(float(squares[0, channel])) if startup else None
        result[f"transient/{name}/increment_startup"] = Endpoint(value, cells if value is not None else 0)
        if len(window) <= 1:
            # Without a shared stored interval there is no window: the increment endpoints are missing and block removal.
            result[f"transient/{name}/increment_peak"] = Endpoint(None, 0)
            result[f"transient/{name}/increment_rms"] = Endpoint(None, 0)
    if len(window) > 1:
        result.update(_window(squares, float(times[window[-1][0]]), cells))
    if reference.completion is None or other.completion is None:
        result[DURATION] = Endpoint(None, 0)
    else:
        result[DURATION] = duration_endpoint(reference.completion, other.completion)
    result[WET_FRACTION_HORIZON] = _horizon_wet_fraction(reference, other, window)
    return result


def endpoints(reference: CaseFields, other: CaseFields) -> dict[str, Endpoint]:
    """Measure every endpoint of the pair's task; both cases must belong to one task."""
    if reference.task != other.task:
        msg = f"Cannot compare a {reference.task} case with a {other.task} case."
        raise ValueError(msg)
    return steady_endpoints(reference, other) if reference.task == "steady_flow" else transient_endpoints(reference, other)


def envelope_endpoints(reference: CaseFields, other: CaseFields) -> dict[str, Endpoint]:
    """
    Measure every endpoint of the pair plus, for a transient pair, the increment endpoints at every shared window end.

    A tolerance table and a replicate envelope need every window end, because each comparison is judged over its own
    shared stored window.
    """
    result = endpoints(reference, other)
    if reference.task == "transient_drying":
        times, ref_states, _other_states = _trajectories(reference, other)
        window = _shared_window(reference, other)
        squares = _increment_squares(reference, other, window)
        cells = int(np.prod(ref_states.shape[-2:]))
        for count in range(1, len(window)):
            result.update(_window(squares[:count], float(times[window[count][0]]), cells))
    return result
