"""Select a visible analysis device and reduce bounded physical error batches.

Responsibilities:
  - Resolve workspace CPU, auto and process-visible logical CUDA preferences
  - Reduce full-support physical error sums through bounded CPU or CUDA batches
  - Report transfer-aware execution evidence and explicit auto resource fallback

Design principles:
  - Only Slurm-visible logical GPU indexes are offered
  - Float64 reductions preserve exact caller-declared spatial support
  - Device execution evidence never becomes scientific identity

This module does NOT:
  - Change CUDA visibility, request cluster allocations or choose physical GPUs
  - Fit preprocessing, infer absent spatial support or render figures
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakSet

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

ComputeBackend = Literal["cpu", "cuda"]
_AUTO_MIN_ELEMENTS = 3_200_000  # 8 cases x 4 fields x 251 x 401 in the bounded profile.
_MAX_CHUNK_INPUT_BYTES = 64 * 1024 * 1024
_BATCH_RANK = 3


@dataclass(frozen=True, slots=True)
class AnalysisDevice:
    """One resolved, process-visible analysis route; never a scientific identity."""

    requested: str
    backend: ComputeBackend
    logical_device: str
    reason: str

    @property
    def label(self) -> str:
        """Disclose both the user's request and the actual execution route."""
        return f"requested {self.requested}; resolved {self.logical_device} ({self.reason})"


class AnalysisComputePreference:
    """One workspace-owned device choice shared by mounted plot callbacks."""

    def __init__(self, requested: str = "auto") -> None:
        """Admit the initial route without changing artifact or metric identity."""
        resolve_analysis_device(requested)
        self.requested = requested
        self._listeners: WeakSet[Callable[[], None]] = WeakSet()

    def set_requested(self, requested: str) -> AnalysisDevice:
        """Validate and publish one change without remounting field selectors."""
        resolved = resolve_analysis_device(requested)
        if requested != self.requested:
            self.requested = requested
            for listener in tuple(self._listeners):
                listener()
        return resolved

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Observe the selected device while the owning viewer remains alive."""
        self._listeners.add(callback)


@dataclass(frozen=True, slots=True)
class ErrorBatchSummary:
    """Full-grid sufficient statistics plus transfer-aware execution evidence."""

    absolute_sum: np.ndarray
    signed_sum: np.ndarray
    squared_sum: np.ndarray
    support_count: np.ndarray
    device: AnalysisDevice


def _torch_cuda() -> Any:
    """Import torch only when the selected route may need CUDA."""
    import torch  # noqa: PLC0415

    return torch


def visible_cuda_options() -> tuple[str, ...]:
    """Offer only process-visible logical indexes under a Slurm allocation."""
    if "SLURM_JOB_ID" not in os.environ:
        return ()
    torch = _torch_cuda()
    if not torch.cuda.is_available():
        return ()
    return tuple(f"cuda:{index}" for index in range(torch.cuda.device_count()))


def selection_status(requested: str) -> str:
    """Report a provisional auto choice or an admitted explicit route."""
    if requested == "auto":
        return "requested auto; resolved per eligible workload (CPU below threshold, CUDA for large allocated batches)"
    return resolve_analysis_device(requested).label


def resolve_analysis_device(requested: str, *, elements: int = 0) -> AnalysisDevice:
    """Resolve cpu/auto/cuda/visible cuda:N without changing device visibility."""
    if requested == "cpu":
        return AnalysisDevice(requested, "cpu", "cpu", "explicit CPU")
    if requested not in {"auto", "cuda"} and not (requested.startswith("cuda:") and requested[5:].isdigit()):
        msg = "Evaluation compute device must be cpu, auto, cuda, or a visible cuda:N."
        raise ValueError(msg)
    if isinstance(elements, bool) or not isinstance(elements, int) or elements < 0:
        msg = "Evaluation workload elements must be a nonnegative integer."
        raise ValueError(msg)
    if requested == "auto" and elements < _AUTO_MIN_ELEMENTS:
        return AnalysisDevice(requested, "cpu", "cpu", f"below measured {_AUTO_MIN_ELEMENTS}-element GPU batch threshold")
    if "SLURM_JOB_ID" not in os.environ:
        if requested == "auto":
            return AnalysisDevice(requested, "cpu", "cpu", "no Slurm GPU allocation in this process")
        msg = "Explicit CUDA Evaluation requires a Slurm-allocated GPU process."
        raise RuntimeError(msg)
    torch = _torch_cuda()
    if not torch.cuda.is_available():
        if requested == "auto":
            return AnalysisDevice(requested, "cpu", "cpu", "no visible accessible CUDA device")
        msg = "Explicit CUDA Evaluation requires a visible accessible CUDA device."
        raise RuntimeError(msg)
    index = 0 if requested in {"auto", "cuda"} else int(requested[5:])
    count = torch.cuda.device_count()
    if index >= count:
        msg = f"Requested logical cuda:{index} is outside {count} visible CUDA device(s)."
        raise ValueError(msg)
    logical = f"cuda:{index}"
    try:
        with torch.inference_mode():
            probe = torch.ones(1, dtype=torch.float64, device=logical)
            probe = probe + probe
            torch.cuda.synchronize(index)
        del probe
    except (RuntimeError, AssertionError) as error:
        if requested == "auto" and isinstance(error, RuntimeError) and _recoverable_cuda_resource_error(error):
            return AnalysisDevice(requested, "cpu", "cpu", f"visible fallback after CUDA resource failure: {type(error).__name__}")
        mode_label = "Auto CUDA" if requested == "auto" else "Explicit CUDA"
        msg = f"{mode_label} Evaluation cannot execute float64 on {logical}: {error}"
        raise RuntimeError(msg) from error
    return AnalysisDevice(requested, "cuda", logical, "visible float64 error reduction")


def _validate_batch(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Require one aligned finite-on-support physical batch."""
    actual = np.asarray(reference)
    predicted = np.asarray(prediction)
    admitted = np.asarray(support)
    if (
        actual.shape != predicted.shape
        or actual.shape != admitted.shape
        or actual.ndim != _BATCH_RANK
        or not actual.size
        or actual.dtype != np.float64
        or predicted.dtype != np.float64
        or admitted.dtype != np.bool_
    ):
        msg = "Error batches require aligned nonempty [case,y,x] float64 states and boolean support."
        raise ValueError(msg)
    if not np.isfinite(actual[admitted]).all() or not np.isfinite(predicted[admitted]).all():
        msg = "Error batch has non-finite reference or prediction on required support."
        raise ValueError(msg)
    return actual, predicted, admitted


def _cpu_error_batch(actual: np.ndarray, predicted: np.ndarray, admitted: np.ndarray, device: AnalysisDevice) -> ErrorBatchSummary:
    """Reduce in float64 while excluding only explicitly absent spatial support."""
    difference = predicted - actual
    if not np.isfinite(difference[admitted]).all():
        msg = "Computed error is non-finite on required support."
        raise ValueError(msg)
    signed = np.where(admitted, difference, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        absolute_sum = np.abs(signed).sum(axis=0, dtype=np.float64)
        signed_sum = signed.sum(axis=0, dtype=np.float64)
        squared_sum = np.square(signed, dtype=np.float64).sum(axis=0, dtype=np.float64)
    counts = admitted.sum(axis=0, dtype=np.int64)
    if not all(np.isfinite(values[counts > 0]).all() for values in (absolute_sum, signed_sum, squared_sum)):
        msg = "Computed CPU error summary is non-finite on required support."
        raise ValueError(msg)
    return ErrorBatchSummary(absolute_sum, signed_sum, squared_sum, counts, device)


def _cuda_error_batch(actual: np.ndarray, predicted: np.ndarray, admitted: np.ndarray, device: AnalysisDevice) -> ErrorBatchSummary:
    """Transfer bounded chunks, reduce on one logical GPU, return summaries only."""
    torch = _torch_cuda()
    cells = actual.shape[1] * actual.shape[2]
    bytes_per_case = cells * (8 + 8 + 1)
    chunk_cases = max(1, _MAX_CHUNK_INPUT_BYTES // bytes_per_case)
    absolute_sum = np.zeros(actual.shape[1:], dtype=np.float64)
    signed_sum = np.zeros_like(absolute_sum)
    squared_sum = np.zeros_like(absolute_sum)
    support_count = np.zeros(actual.shape[1:], dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, actual.shape[0], chunk_cases):
            end = min(start + chunk_cases, actual.shape[0])
            mask = torch.from_numpy(np.ascontiguousarray(admitted[start:end])).to(device.logical_device)
            ref = torch.from_numpy(np.ascontiguousarray(actual[start:end])).to(device.logical_device)
            pred = torch.from_numpy(np.ascontiguousarray(predicted[start:end])).to(device.logical_device)
            difference = pred - ref
            signed = torch.where(mask, difference, 0.0)
            absolute_sum += torch.abs(signed).sum(dim=0).cpu().numpy()
            signed_sum += signed.sum(dim=0).cpu().numpy()
            squared_sum += torch.square(signed).sum(dim=0).cpu().numpy()
            support_count += mask.sum(dim=0).cpu().numpy()
        torch.cuda.synchronize(int(device.logical_device[5:]))
    if not all(np.isfinite(values[support_count > 0]).all() for values in (absolute_sum, signed_sum, squared_sum)):
        msg = "Computed CUDA error summary is non-finite on required support."
        raise ValueError(msg)
    return ErrorBatchSummary(absolute_sum, signed_sum, squared_sum, support_count, device)


def _recoverable_cuda_resource_error(error: RuntimeError) -> bool:
    """Recognize only known allocation failures eligible for auto CPU fallback."""
    return "out of memory" in str(error).lower() or "cuda error: allocation" in str(error).lower()


def reduce_error_batch(reference: np.ndarray, prediction: np.ndarray, support: np.ndarray, *, requested: str = "auto") -> ErrorBatchSummary:
    """Return exact-support full-grid sums; auto resource fallback is visible."""
    actual, predicted, admitted = _validate_batch(reference, prediction, support)
    device = resolve_analysis_device(requested, elements=actual.size)
    if device.backend == "cpu":
        return _cpu_error_batch(actual, predicted, admitted, device)
    try:
        return _cuda_error_batch(actual, predicted, admitted, device)
    except RuntimeError as error:
        if requested != "auto" or not _recoverable_cuda_resource_error(error):
            raise
        fallback = AnalysisDevice(requested, "cpu", "cpu", f"visible fallback after CUDA resource failure: {type(error).__name__}")
        return _cpu_error_batch(actual, predicted, admitted, fallback)
