"""Measure Train-fitted scaled increments, the Huber regime, and stored time steps.

Responsibilities:
  - Fit the transient scaling on the selected membership through the maintained scaling service
  - Accumulate per-channel RMS, quantiles, exact zeros, the fraction above the Huber beta, and equal-weight loss shares
  - Tabulate the stored time steps of every case with a uniformity flag
  - Scan transient reference cases once for the spectral, stride, increment, and time-step tables

Design principles:
  - Scaling, increments, and stride views reuse the training services unchanged
  - Every count is exact; only the quantiles come from a fixed logarithmic histogram
  - Statistics describe targets only: they are priors for the Huber and channel-weight ablations

This module does NOT:
  - Persist a scaling artifact or reuse a run's scaling cache
  - Choose the loss, its beta, or channel weights

Notes:
  Huber term per element: torch huber_loss with delta = beta, as the transient loss applies it.
  The equal-weight loss share of a channel is its mean Huber value of the scaled
  increment, i.e. the loss of a zero-increment prediction, over the sum of all channels.
  Quantiles report the upper edge of the histogram bin that reaches them (100 bins per
  decade, relative width 2.3 %); values below the lowest edge report 0.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from src import common, datasets
from src.learning.transient import learning_transient_contracts as transient_contracts
from src.learning.transient import learning_transient_scaling as transient_scaling

from . import component_checks_sources as sources
from . import component_checks_spectral as spectral

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

    from .component_checks_sources import Membership, TransientReferenceCase

QUANTILES: Final = (0.5, 0.9, 0.99)
DT_TOLERANCE_HOURS: Final = 1e-6
_HISTOGRAM_EDGES: Final = torch.logspace(-12, 6, 1801, dtype=torch.float64)
_TASK_ID: Final = "transient_drying"


@dataclass(frozen=True, slots=True)
class TransientComponentTables:
    """
    Hold the tables of one pass over transient reference cases.

    Attributes
    ----------
    spectral : pandas.DataFrame
        :func:`component_checks_spectral.transient_truncation_errors` of every case.
    stride : pandas.DataFrame
        :func:`component_checks_spectral.stride_floor` of every case.
    increments : pandas.DataFrame
        :func:`scaled_increment_statistics` over all cases.
    time_steps : pandas.DataFrame
        :func:`time_step_table` of every case.

    """

    spectral: pd.DataFrame
    stride: pd.DataFrame
    increments: pd.DataFrame
    time_steps: pd.DataFrame


def fit_membership_scaling(
    dataset_id: str,
    *,
    membership: Membership,
    config: Mapping[str, Any],
    storage_root: Path | str | None = None,
    max_cases: int | None = None,
) -> transient_scaling.TransientScalingArtifact:
    """
    Fit the transient scaling on one membership exactly as training fits it on Train.

    Parameters
    ----------
    dataset_id : str
        Published transient Dataset identifier.
    membership : {"train", "validation"}
        Partition role whose leading cases are fitted.
    config : Mapping[str, Any]
        Maintained transient learning configuration; ``data.spatial_stride``,
        ``scaling.mode``, ``input_profile`` and ``temporal`` are read.
    storage_root : pathlib.Path or str, optional
        Scientific storage root; defaults to the canonical resolution.
    max_cases : int, optional
        Leading membership cases to fit; all when omitted.

    Returns
    -------
    TransientScalingArtifact
        In-memory scaling; its membership digest hashes the fitted case IDs.

    """
    factory = datasets.runtime.factory
    dataset = factory.create_dataset(
        factory.DatasetRequest(
            dataset_id=dataset_id,
            dataset_view=_TASK_ID,
            transient_sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
            storage_root=storage_root,
            spatial_stride=config["data"]["spatial_stride"],
        )
    )
    if not isinstance(dataset, datasets.runtime.transient.TransientPTShardDataset):
        msg = f"Dataset {dataset_id!r} did not resolve to a transient PT-shard runtime."
        raise TypeError(msg)
    try:
        binding = dataset.manifest_binding
        partition = None if binding is None else binding.case_partition
        case_ids = sources.select_membership_cases(partition, membership, dataset_id=dataset_id, max_cases=max_cases)
        identity = datasets.preprocessing.splits.split_dataset_identity(binding, dataset.payload_identity) if binding else {}
        # The selection rebuilds the same runtime class, so it stays a PT-shard runtime.
        selected = cast("datasets.runtime.transient.TransientPTShardDataset", datasets.runtime.transient.select_transient_cases(dataset, case_ids))
    except BaseException:
        dataset.close()
        raise
    try:
        tensorizer = transient_contracts.TransientTensorizerSpec.from_mapping(
            {"input_profile": config["input_profile"], "temporal_conditioning": config["temporal"]["temporal_conditioning"]}
        )
        return transient_scaling.fit_transient_scaling(
            selected.iter_scaling_items(),
            tensorizer=tensorizer,
            dataset_identity=identity,
            train_membership_digest=common.serialization.canonical_json_sha256(list(case_ids)),
            horizon=selected.configured_regular_horizon,
            scale_mode=config["scaling"]["mode"],
        )
    finally:
        selected.close()


def scaled_increment_statistics(batches: Iterable[Tensor], *, channels: Sequence[str], beta: float) -> pd.DataFrame:
    """
    Accumulate per-channel statistics of scaled increments ``[transition, channel, y, x]``.

    Returns
    -------
    pandas.DataFrame
        Index ``channel``; columns ``count``, ``rms``, ``q50``, ``q90``, ``q99``
        and ``max`` of the magnitude, ``zero_fraction``,
        ``above_beta_fraction`` (strictly above ``beta``) and ``loss_share``.

    """
    size = len(channels)
    count = np.zeros(size)
    square, zeros, above, huber, peak = (np.zeros(size) for _ in range(5))
    histogram = np.zeros((size, len(_HISTOGRAM_EDGES) + 1))
    for batch in batches:
        values = batch.movedim(1, 0).reshape(size, -1).to(torch.float64)
        magnitude = values.abs()
        count += values.shape[1]
        square += values.square().sum(dim=1).numpy()
        zeros += (values == 0).sum(dim=1).numpy()
        above += (magnitude > beta).sum(dim=1).numpy()
        huber += functional.huber_loss(values, torch.zeros_like(values), reduction="none", delta=beta).sum(dim=1).numpy()
        peak = np.maximum(peak, magnitude.amax(dim=1).numpy())
        bins = torch.bucketize(magnitude, _HISTOGRAM_EDGES)
        histogram += np.stack([torch.bincount(row, minlength=histogram.shape[1]).numpy() for row in bins])
    # Upper edge of every bucketize bin; the lowest bin (zeros and values below 1e-12) reports 0.
    upper = np.concatenate(([0.0], _HISTOGRAM_EDGES.numpy()[1:], [np.inf]))
    cumulative = histogram.cumsum(axis=1) / count[:, None]
    table = pd.DataFrame({"count": count, "rms": np.sqrt(square / count)}, index=pd.Index(channels, name="channel"))
    for quantile in QUANTILES:
        reached = np.argmax(cumulative >= quantile, axis=1)
        table[f"q{round(100 * quantile)}"] = np.minimum(upper[reached], peak)
    table["max"] = peak
    table["zero_fraction"] = zeros / count
    table["above_beta_fraction"] = above / count
    mean_huber = huber / count
    table["loss_share"] = mean_huber / mean_huber.sum() if mean_huber.sum() > 0 else np.nan
    return table


def time_step_table(times: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """
    Tabulate stored time steps per case and over all cases.

    Returns
    -------
    pandas.DataFrame
        Rows per case and a last ``all cases`` row with ``steps``, ``dt_min_h``,
        ``dt_median_h``, ``dt_max_h`` and ``uniform`` (spread within :data:`DT_TOLERANCE_HOURS`).

    """
    steps = {label: np.diff(np.asarray(values, dtype=np.float64)) for label, values in times.items()}
    steps["all cases"] = np.concatenate(list(steps.values())) if steps else np.zeros(0)
    rows = [
        {
            "case": label,
            "steps": int(values.size),
            "dt_min_h": float(values.min()) if values.size else np.nan,
            "dt_median_h": float(np.median(values)) if values.size else np.nan,
            "dt_max_h": float(values.max()) if values.size else np.nan,
            "uniform": bool(values.size) and float(values.max() - values.min()) <= DT_TOLERANCE_HOURS,
        }
        for label, values in steps.items()
    ]
    return pd.DataFrame(rows).set_index("case")


def scan_transient_cases(
    cases: Iterable[TransientReferenceCase],
    *,
    scaling: transient_scaling.TransientScalingArtifact,
    spatial_stride: int,
    beta: float,
    settings: pd.DataFrame,
) -> TransientComponentTables:
    """
    Evaluate the spectral, stride, increment, and time-step checks in one pass over cases.

    Parameters
    ----------
    cases : Iterable[TransientReferenceCase]
        Membership cases, e.g. from :func:`component_checks_sources.iter_transient_reference_cases`.
    scaling : TransientScalingArtifact
        Scaling fitted on the same membership.
    spatial_stride : int
        Training stride of the spectral and increment views.
    beta : float
        Maintained Huber transition scale.
    settings : pandas.DataFrame
        :func:`component_checks_spectral.maintained_mode_settings` of the task.

    """
    spectral_rows: list[pd.DataFrame] = []
    stride_rows: list[pd.DataFrame] = []
    times: dict[str, np.ndarray] = {}

    def scaled() -> Iterable[Tensor]:
        """Evaluate the per-case tables and stream each case's scaled increments."""
        for case in cases:
            spectral_rows.append(spectral.transient_truncation_errors(case, scaling=scaling, spatial_stride=spatial_stride, settings=settings))
            stride_rows.append(spectral.stride_floor(case))
            times[case.label] = case.time_hours
            states = spectral.strided_states(case, spatial_stride).to(scaling.dtype)
            yield scaling.encode_delta(states[1:] - states[:-1])

    channels = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.dynamic_state)
    increments = scaled_increment_statistics(scaled(), channels=channels, beta=beta)
    return TransientComponentTables(
        spectral=pd.concat(spectral_rows, ignore_index=True),
        stride=pd.concat(stride_rows, ignore_index=True),
        increments=increments,
        time_steps=time_step_table(times),
    )
