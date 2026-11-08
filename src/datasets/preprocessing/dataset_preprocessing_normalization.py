"""Own steady per-channel normalizer fitting, reconstruction and preprocessing specification.

Responsibilities:
  - Fit per-channel preprocessing from training membership only
  - Reconstruct processors from isolated persisted tensors
  - State the steady preprocessing specification that the shared scaling identity binds

Design principles:
  - Saved tensors are validated on CPU before use
  - Zero-variance channels retain the established denominator floor

This module does NOT:
  - Choose split membership or construct training DataLoaders
  - Cache, envelope or persist scaling artifacts; the shared scaling service owns that
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from torch import Tensor

if TYPE_CHECKING:
    from src.domain.tasks.domain_task_spec import TaskSpec


_REQUIRED_NORMALIZER_STATE_KEYS = (
    "in_normalizer.mean",
    "in_normalizer.std",
    "out_normalizer.mean",
    "out_normalizer.std",
)
_NORMALIZER_STATE_RANK = 4
_NORMALIZER_DENOMINATOR_FLOOR = 1e-7


def data_processor_from_state(
    state: Mapping[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> DefaultDataProcessor:
    """
    Reconstruct a data processor from persisted normalizer tensors.

    Parameters
    ----------
    state : Mapping[str, Any]
        Exact four-key normalizer state. Means and standard deviations must be
        finite real tensors shaped ``[1, channels, 1, 1]``. Standard deviations
        may be zero because the processor applies its denominator floor.
    device : torch.device | str, optional
        Target processor device for isolated state tensors.

    Returns
    -------
    DefaultDataProcessor
        Processor containing detached, cloned input/output normalization state
        on ``device`` with normalization axes ``[0, 2, 3]``.

    Raises
    ------
    TypeError
        If the state/key tensors have incompatible container, dtype, or layout
        types.
    ValueError
        If keys, shapes, finiteness, mean/std pairing, or non-negative standard
        deviations violate the persisted normalizer contract.

    """
    if not isinstance(state, Mapping):
        msg = "Saved normalizer state must be a mapping."
        raise TypeError(msg)
    missing = sorted(set(_REQUIRED_NORMALIZER_STATE_KEYS).difference(state))
    unexpected = sorted(set(state).difference(_REQUIRED_NORMALIZER_STATE_KEYS))
    if missing or unexpected:
        msg = f"Saved normalizer state keys do not match. Missing: {missing}. Unexpected: {unexpected}."
        raise ValueError(msg)

    tensors: dict[str, Tensor] = {}
    for key in _REQUIRED_NORMALIZER_STATE_KEYS:
        value = state.get(key)
        if not isinstance(value, Tensor):
            msg = f"Saved normalizer state {key!r} must be a torch.Tensor."
            raise TypeError(msg)
        if not value.is_floating_point() or value.is_complex():
            msg = f"Saved normalizer state {key!r} must be a real floating-point tensor."
            raise TypeError(msg)
        if value.ndim != _NORMALIZER_STATE_RANK or value.shape[0] != 1 or value.shape[2:] != (1, 1):
            msg = f"Saved normalizer state {key!r} must have shape (1, channels, 1, 1), got {tuple(value.shape)}."
            raise ValueError(msg)
        tensors[key] = value

    if tensors["in_normalizer.mean"].shape != tensors["in_normalizer.std"].shape:
        msg = "Saved input normalizer mean/std shapes do not match."
        raise ValueError(msg)
    if tensors["out_normalizer.mean"].shape != tensors["out_normalizer.std"].shape:
        msg = "Saved output normalizer mean/std shapes do not match."
        raise ValueError(msg)
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        msg = "Saved normalizer state contains non-finite values."
        raise ValueError(msg)
    if torch.any(tensors["in_normalizer.std"] < 0) or torch.any(tensors["out_normalizer.std"] < 0):
        msg = "Saved normalizer standard deviations must be non-negative."
        raise ValueError(msg)
    target_device = torch.device(device)
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3], eps=_NORMALIZER_DENOMINATOR_FLOOR)
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3], eps=_NORMALIZER_DENOMINATOR_FLOOR)
    processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )
    in_normalizer.mean = tensors["in_normalizer.mean"].detach().clone().to(target_device)
    in_normalizer.std = tensors["in_normalizer.std"].detach().clone().to(target_device)
    out_normalizer.mean = tensors["out_normalizer.mean"].detach().clone().to(target_device)
    out_normalizer.std = tensors["out_normalizer.std"].detach().clone().to(target_device)
    processor.device = target_device
    return processor


def preprocessing_spec(task: TaskSpec) -> dict[str, Any]:
    """Return the steady preprocessing specification bound into the scaling identity."""
    return {
        "kind": "unit_gaussian_per_channel",
        "normalization_axes": list(task.normalization_axes),
        "denominator_floor": _NORMALIZER_DENOMINATOR_FLOOR,
        "in_channels": task.in_channels,
        "out_channels": task.out_channels,
    }


def processor_state(data_processor: DefaultDataProcessor) -> dict[str, Tensor]:
    """Return isolated CPU normalizer tensors after validating that they reconstruct a processor."""
    state = {key: value.detach().cpu().clone() for key, value in data_processor.state_dict().items()}
    data_processor_from_state(state, device="cpu")
    return state


def fit_data_processor(
    batches: Iterable[Mapping[str, Any]],
    *,
    task: TaskSpec,
) -> DefaultDataProcessor:
    """Fit task normalizers only from the supplied training membership."""
    xs_train: list[Tensor] = []
    ys_train: list[Tensor] = []
    for batch in batches:
        xs_train.append(batch["x"])
        ys_train.append(batch["y"])
    x_train = torch.cat(xs_train, dim=0)
    y_train = torch.cat(ys_train, dim=0)
    normalization_axes = list(task.normalization_axes)
    in_normalizer = UnitGaussianNormalizer(dim=normalization_axes, eps=_NORMALIZER_DENOMINATOR_FLOOR)
    in_normalizer.fit(x_train)
    out_normalizer = UnitGaussianNormalizer(dim=normalization_axes, eps=_NORMALIZER_DENOMINATOR_FLOOR)
    out_normalizer.fit(y_train)
    return DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )
