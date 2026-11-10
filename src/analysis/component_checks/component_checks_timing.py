"""Time the steady physics-loss path per derivative backend and device.

Responsibilities:
  - Time forward plus backward of the dimensionless steady loss terms per backend
  - Record peak memory, device name, thread count, and TF32 state with every row
  - Report a device that is not available as unavailable instead of substituting another

Design principles:
  - The domain evaluator and the maintained derivative backends are timed unchanged
  - Warm-ups precede timed repeats; CUDA work is synchronized around each repeat
  - Measurements are node-specific and returned, never persisted

This module does NOT:
  - Time model forward passes, data loading, or optimizer steps
  - Choose a backend or change a configuration
"""

from __future__ import annotations

import functools
import resource
import statistics
import time
from typing import TYPE_CHECKING, Final

import pandas as pd
import torch

from src import domain

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from .component_checks_steady import DerivativeBackend

DEFAULT_WARMUPS: Final = 3
DEFAULT_REPEATS: Final = 5
_BYTES_PER_MIB: Final = 1024.0**2
_RSS_KIB_PER_MIB: Final = 1024.0


def _loss_step(
    inputs: Tensor, outputs: Tensor, *, input_fields: Sequence[str], output_fields: Sequence[str], backend: DerivativeBackend, crop: int
) -> None:
    """Run forward and backward of the summed dimensionless loss terms once."""
    prediction = outputs.detach().clone().requires_grad_()
    terms = domain.physics.brinkman.evaluate_steady_2d_brinkman(
        inputs,
        prediction,
        input_fields=input_fields,
        output_fields=output_fields,
        derivatives=backend.build(),
        boundary=domain.physics.contracts.PRESSURE_BOUNDARY_KIND,
        interior_crop=crop,
    )
    (terms.momentum_mse + terms.continuity_mse + terms.pressure_inlet_mse + terms.pressure_outlet_mean_square).backward()


def _synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work so wall time covers the kernels."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_physics_loss(
    inputs: Tensor,
    outputs: Tensor,
    *,
    input_fields: Sequence[str],
    output_fields: Sequence[str],
    backends: Sequence[DerivativeBackend],
    crop: int,
    warmups: int = DEFAULT_WARMUPS,
    repeats: int = DEFAULT_REPEATS,
) -> pd.DataFrame:
    """
    Time forward plus backward of the steady physics loss per backend on the CPU and, when visible, the GPU.

    Parameters
    ----------
    inputs, outputs : torch.Tensor
        Physical float32 BCHW batch, e.g. reference cases tiled to the training batch size.
    input_fields, output_fields : Sequence[str]
        Task channel declarations.
    backends : Sequence[DerivativeBackend]
        Derivative configurations to time.
    crop : int
        Interior crop of the momentum term.
    warmups, repeats : int, optional
        Untimed and timed iterations; the defaults are the minimum for a reported cost.

    Returns
    -------
    pandas.DataFrame
        One row per device (``cpu``, ``cuda``) and backend; CUDA rows say
        ``unavailable`` without a visible GPU. Columns ``status``, ``device_name``,
        ``threads``, ``tf32`` (matmul and cuDNN flags), ``batch_shape``,
        ``median_ms``, ``peak_memory_mib`` and ``memory_measure``. CPU peak
        memory is the growth of the process maximum resident set over warm-ups
        and repeats of one backend, a lower bound that stays 0 when earlier work
        in the process already reached that peak.

    """
    rows: list[dict[str, object]] = []
    tf32 = f"matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}"
    for name in ("cpu", "cuda"):
        common = {"device": name, "threads": torch.get_num_threads(), "tf32": tf32, "batch_shape": tuple(outputs.shape)}
        if name == "cuda" and not torch.cuda.is_available():
            rows.extend({**common, "backend": backend.label, "status": "unavailable", "device_name": "unavailable"} for backend in backends)
            continue
        device = torch.device(name)
        device_name = torch.cuda.get_device_name(device) if name == "cuda" else "cpu"
        batch_inputs, batch_outputs = inputs.to(device), outputs.to(device)
        for backend in backends:
            step = functools.partial(
                _loss_step, batch_inputs, batch_outputs, input_fields=input_fields, output_fields=output_fields, backend=backend, crop=crop
            )
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            for _ in range(warmups):
                step()
            _synchronize(device)
            if name == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            durations = []
            for _ in range(repeats):
                _synchronize(device)
                started = time.perf_counter()
                step()
                _synchronize(device)
                durations.append(1000.0 * (time.perf_counter() - started))
            if name == "cuda":
                peak, measure = torch.cuda.max_memory_allocated(device) / _BYTES_PER_MIB, "cuda_max_allocated"
            else:
                peak = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss_before) / _RSS_KIB_PER_MIB
                measure = "process_max_rss_growth"
            rows.append(
                {
                    **common,
                    "backend": backend.label,
                    "status": "measured",
                    "device_name": device_name,
                    "median_ms": statistics.median(durations),
                    "peak_memory_mib": peak,
                    "memory_measure": measure,
                }
            )
    return pd.DataFrame(rows)
