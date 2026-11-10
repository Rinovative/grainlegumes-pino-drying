"""Measure Fourier-mode truncation floors and spatial-stride floors of reference fields.

Responsibilities:
  - Keep exactly the Fourier coefficients a neuraloperator spectral layer keeps per axis
  - Apply the models' symmetric zero domain padding before truncation and remove it after
  - Tabulate relative and edge-band truncation errors over mode counts, paddings, and maintained mode pairs
  - Inventory the maintained mode counts and domain padding of every learning configuration
  - Reconstruct endpoint-preserving stride views bilinearly and report their full-grid error

Design principles:
  - The kept set mirrors neuralop 2.0 SpectralConv; padding mirrors its DomainPadding
  - Results are floors or priors: the networks truncate lifted features, not the raw fields
  - Stride views come from the Dataset contract's index rule, never a second one

This module does NOT:
  - Train, build models, or choose a mode count, padding, or stride
  - Load Datasets or render figures

Notes:
  Kept set of n_modes = (m_y, m_x) on a padded (n_y, n_x) grid, probed against SpectralConv:
    y, full FFT after fftshift: frequencies -(m_y // 2) .. m_y // 2 - 1 + m_y % 2
    x, rFFT on the last axis: indices 0 .. m_x // 2, i.e. m_x // 2 + 1 coefficients
  Padding: round(p n) zero cells per side and axis (DomainPadding, F.pad mode "constant").
  Fields are normalised as in training (steady out-normaliser, transient state encoding and
  increment scale, fitted on the membership), without per-sample centring, so the jump
  from a nonzero sample mean to the zero padding counts; errors are divided by the mean
  square of the normalised channel. The edge band holds the cells within EDGE_CELLS of the
  nearest edge. Stride floors divide the per-frame RMSE by the case's channel standard
  deviation (states) or RMS (increments) over all stored times.

"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from src import datasets, domain
from src.learning.models import learning_models_factory as model_factory

from . import component_checks_sources as sources
from . import component_checks_steady as steady

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from src.learning.transient.learning_transient_scaling import TransientScalingArtifact

    from .component_checks_sources import SteadyReference, TransientReferenceCase

MODE_COUNTS: Final = (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 400)
PADDINGS: Final = (0.0, 0.1)
EDGE_CELLS: Final = 2
TIME_SAMPLES: Final = 16
STRIDES: Final = (1, 2, 4)
ModePair = tuple[int | None, int | None]


def kept_mode_mask(shape: tuple[int, int], modes: ModePair) -> Tensor:
    """
    Return the rFFT coefficients ``[y, x // 2 + 1]`` a spectral layer keeps.

    Parameters
    ----------
    shape : tuple[int, int]
        Padded spatial ``(y, x)`` shape the layer transforms.
    modes : tuple[int | None, int | None]
        ``n_modes`` ``(m_y, m_x)``; ``None`` keeps every coefficient of that axis.

    """
    height, width = shape
    frequencies = torch.fft.fftfreq(height, d=1.0 / height).round()
    keep_y = torch.ones(height, dtype=torch.bool)
    if modes[0] is not None:
        keep_y = (frequencies >= -(modes[0] // 2)) & (frequencies <= modes[0] // 2 - 1 + modes[0] % 2)
    keep_x = torch.arange(width // 2 + 1) <= (width if modes[1] is None else modes[1] // 2)
    return keep_y[:, None] & keep_x[None, :]


def truncate_to_modes(field: Tensor, modes: ModePair, *, padding: float) -> Tensor:
    """Pad ``[..., y, x]`` with zeros, keep the layer's Fourier modes, and crop back to the field."""
    height, width = field.shape[-2:]
    pad_y, pad_x = round(padding * height), round(padding * width)
    padded = functional.pad(field, (pad_x, pad_x, pad_y, pad_y))
    spectrum = torch.fft.rfftn(padded, dim=(-2, -1)) * kept_mode_mask((padded.shape[-2], padded.shape[-1]), modes)
    restored = torch.fft.irfftn(spectrum, s=padded.shape[-2:], dim=(-2, -1))
    return restored[..., pad_y : pad_y + height, pad_x : pad_x + width]


def _relative(error_squared: float, scale_squared: float) -> float:
    """Return ``sqrt(error / scale)``: 0 when nothing is lost, ``inf`` for a loss of a constant field."""
    if scale_squared > 0:
        return math.sqrt(error_squared / scale_squared)
    return 0.0 if error_squared == 0 else math.inf


def truncation_errors(
    fields: Tensor,
    names: Sequence[str],
    *,
    mode_counts: Sequence[int] = MODE_COUNTS,
    paddings: Sequence[float] = PADDINGS,
    mode_pairs: Sequence[tuple[str, tuple[int, int]]] = (),
) -> pd.DataFrame:
    """
    Tabulate truncation errors of normalised fields pooled over their leading dimension.

    Parameters
    ----------
    fields : torch.Tensor
        ``[sample, channel, y, x]`` values normalised as in training and not
        centred per sample; samples are pooled, e.g. the time frames of one case.
        Per-axis curves keep only mode counts below the padded axis size.
    names : Sequence[str]
        Channel names.
    mode_counts : Sequence[int], optional
        Mode counts of the per-axis curves; the other axis keeps every mode.
    paddings : Sequence[float], optional
        Domain-padding fractions per side.
    mode_pairs : Sequence[tuple[str, tuple[int, int]]], optional
        Labelled ``(m_y, m_x)`` pairs, e.g. of maintained configurations.

    Returns
    -------
    pandas.DataFrame
        Rows with ``field``, ``axis`` (``y``, ``x`` or ``config``), ``config``,
        ``modes_y``, ``modes_x`` (``None``: all kept), ``padding``,
        ``relative_l2`` and ``edge_relative`` (RMS error over the whole field
        and over the edge band, each divided by the channel's root mean square).

    """
    work = fields.to(torch.float64)
    shape = (work.shape[-2], work.shape[-1])
    edge = torch.from_numpy(steady.edge_distance(shape) <= EDGE_CELLS)
    scale = work.square().mean(dim=(0, -2, -1))
    rows: list[dict[str, object]] = []
    for padding in paddings:
        height, width = (size + 2 * round(padding * size) for size in shape)
        candidates: list[tuple[str, str | None, ModePair]] = [
            *(("y", None, (count, None)) for count in mode_counts if count < height),
            *(("x", None, (None, count)) for count in mode_counts if count // 2 + 1 < width // 2 + 1),
            *(("config", label, pair) for label, pair in mode_pairs),
        ]
        for axis, label, modes in candidates:
            error = (truncate_to_modes(work, modes, padding=padding) - work).square()
            total = error.mean(dim=(0, -2, -1))
            band = error[..., edge].mean(dim=(0, -1))
            rows.extend(
                {
                    "field": name,
                    "axis": axis,
                    "config": label,
                    "modes_y": modes[0],
                    "modes_x": modes[1],
                    "padding": padding,
                    "relative_l2": _relative(float(total[index]), float(scale[index])),
                    "edge_relative": _relative(float(band[index]), float(scale[index])),
                }
                for index, name in enumerate(names)
            )
    return pd.DataFrame(rows)


def maintained_mode_settings(task: str) -> pd.DataFrame:
    """
    Inventory the ``[Y, X]`` mode counts and domain padding of maintained experiments.

    Returns
    -------
    pandas.DataFrame
        One row per experiment (Optuna studies search their modes): ``config``,
        ``model``, ``modes_y``, ``modes_x`` (FNO/RNO ``n_modes``, U-NO
        ``modes_y``/``modes_x`` of its first layer) and ``domain_padding``,
        each over the model kind's defaults as the architecture label reads them.

    """
    rows: list[dict[str, object]] = []
    for item in sources.iter_maintained_configs(task):
        if item.kind != "experiment":
            continue
        kind = item.config["model"]["kind"]
        params = {**model_factory.model_defaults(kind), **item.config["model"].get("params", {})}
        modes = (params["modes_y"], params["modes_x"]) if kind == "uno" else params["n_modes"]
        rows.append(
            {
                "config": item.name,
                "model": kind,
                "modes_y": int(modes[0]),
                "modes_x": int(modes[1]),
                "domain_padding": params.get("domain_padding"),
            }
        )
    return pd.DataFrame(rows)


def _mode_pairs(settings: pd.DataFrame) -> list[tuple[str, tuple[int, int]]]:
    """Return one labelled ``(m_y, m_x)`` pair per distinct maintained setting."""
    pairs: dict[tuple[int, int], str] = {}
    for model, modes_y, modes_x in zip(settings["model"], settings["modes_y"], settings["modes_x"], strict=True):
        pairs.setdefault((int(modes_y), int(modes_x)), f"{model} {int(modes_y)}x{int(modes_x)}")
    return [(label, pair) for pair, label in pairs.items()]


def steady_truncation_errors(reference: SteadyReference, settings: pd.DataFrame) -> pd.DataFrame:
    """
    Return :func:`truncation_errors` of every steady output field, per case, with the maintained mode pairs.

    The outputs pass through the training out-normaliser fitted on the reference membership.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    processor = datasets.preprocessing.normalization.fit_data_processor([{"x": reference.inputs, "y": reference.outputs}], task=task)
    # fit_data_processor always sets the output normaliser; neuralop types it as optional.
    outputs = cast("Any", processor.out_normalizer).transform(reference.outputs)
    frames = [
        truncation_errors(outputs[index : index + 1], reference.output_fields, mode_pairs=_mode_pairs(settings)).assign(case=label)
        for index, label in enumerate(reference.case_labels)
    ]
    return pd.concat(frames, ignore_index=True)


def stride_view(field: Tensor, stride: int) -> Tensor:
    """Return the endpoint-preserving ``[..., y, x]`` view the transient runtime builds for ``spatial_stride``."""
    representation = datasets.contracts.transient.resolve_spatial_representation((field.shape[-2], field.shape[-1]), stride)
    return field[..., list(representation.y_indices), :][..., list(representation.x_indices)]


def strided_states(case: TransientReferenceCase, spatial_stride: int) -> Tensor:
    """Return the float32 case states ``[time, channel, y, x]`` in one stride view."""
    return torch.stack([stride_view(torch.from_numpy(case.states[name]), spatial_stride) for name in case.states], dim=1).to(torch.float32)


def transient_truncation_errors(
    case: TransientReferenceCase, *, scaling: TransientScalingArtifact, spatial_stride: int, settings: pd.DataFrame
) -> pd.DataFrame:
    """
    Return :func:`truncation_errors` of states and increments at the training stride for one case.

    At most :data:`TIME_SAMPLES` evenly spaced stored times are pooled; the
    increments are the transitions starting at those times. States pass through
    the scaling's state encoding and increments through its increment scale.
    """
    states = strided_states(case, spatial_stride).to(scaling.dtype)
    picks = np.unique(np.linspace(0, states.shape[0] - 2, min(TIME_SAMPLES, states.shape[0] - 1)).round().astype(int))
    names = tuple(case.states)
    pairs = _mode_pairs(settings)
    encoded_states = scaling.encode_state(states[picks])
    encoded_increments = scaling.encode_delta(states[picks + 1] - states[picks])
    return pd.concat(
        [
            truncation_errors(encoded_states, names, mode_pairs=pairs).assign(quantity="state"),
            truncation_errors(encoded_increments, names, mode_pairs=pairs).assign(quantity="increment"),
        ],
        ignore_index=True,
    ).assign(case=case.label)


def stride_floor(case: TransientReferenceCase) -> pd.DataFrame:
    """
    Tabulate the bilinear full-grid reconstruction error of each :data:`STRIDES` view of one case.

    Returns
    -------
    pandas.DataFrame
        Rows with ``case``, ``stride``, ``quantity`` (``state`` or
        ``increment``), ``field``, ``status`` and ``max_normalized_rmse``, the
        maximum over stored times. A stride whose view would drop a grid
        endpoint is refused by the Dataset contract and reported with NaN.

    Notes
    -----
    The floor applies only to evaluation on the full grid.

    """
    shape = next(iter(case.states.values())).shape[-2:]
    quantities = ("state", "increment")
    rows: list[dict[str, object]] = []
    for stride in STRIDES:
        try:
            representation = datasets.contracts.transient.resolve_spatial_representation((shape[0], shape[1]), stride)
        except ValueError as error:
            refused = {"case": case.label, "stride": stride, "status": f"refused: {error}", "max_normalized_rmse": math.nan}
            rows.extend({**refused, "quantity": quantity, "field": name} for quantity in quantities for name in case.states)
            continue
        for name, values in case.states.items():
            states = torch.from_numpy(values)[:, None]
            for quantity in quantities:
                channel = states if quantity == "state" else states[1:] - states[:-1]
                view = channel[..., list(representation.y_indices), :][..., list(representation.x_indices)]
                rebuilt = view if stride == 1 else functional.interpolate(view, size=shape, mode="bilinear", align_corners=True)
                rmse = (rebuilt - channel).square().mean(dim=(-3, -2, -1)).sqrt()
                scale = float(channel.std() if quantity == "state" else channel.square().mean().sqrt())
                rows.append(
                    {
                        "case": case.label,
                        "stride": stride,
                        "quantity": quantity,
                        "field": name,
                        "status": "admissible",
                        "max_normalized_rmse": _relative(float(rmse.max()) ** 2, scale**2),
                    }
                )
    return pd.DataFrame(rows)
