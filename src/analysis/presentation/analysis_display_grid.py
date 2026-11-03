"""Admit exact spatial comparisons and sample presentation endpoints.

Responsibilities:
  - Bind spatial grid identity to coordinates, source indexes and field order
  - Reject direct comparisons with incompatible spatial representation
  - Produce deterministic endpoint-preserving display indexes and samples

Design principles:
  - Presentation sampling never changes full-grid numerical reductions
  - Both physical endpoints remain present in sampled maps
  - Caller arrays and canonical grid identities remain unchanged

This module does NOT:
  - Interpolate scientific fields or reconcile incompatible grids
  - Select fields, load artifacts or calculate numerical error metrics
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Literal

import numpy as np

DisplayStride = Literal[1, 2, 4, "auto"]
_SPATIAL_AXES = 2
_COORDINATE_RANK = 3
_MINIMUM_AXIS_LENGTH = 1
_AUTO_MEDIUM_AXIS = 400
_AUTO_LARGE_AXIS = 800
COMMON_GRID_ARTIFACT_COMMAND = (
    "./scripts/slurm_ml.sh --mode gpu --partition gpu --cpus-per-task 4 "
    "--mem 32G --time 02:00:00 --gres gpu:rtx6000ada:1 artifacts "
    "--run-dir <run-dir> --evaluation-spatial-stride 1"
)


@dataclass(frozen=True, slots=True)
class SpatialGridIdentity:
    """Identify coordinates and the declared artifact spatial representation."""

    shape: tuple[int, int]
    coordinate_digest: str
    coordinate_units: tuple[str, str]
    artifact_stride: int
    source_shape: tuple[int, int]
    axis_indices: tuple[tuple[int, ...], tuple[int, ...]]
    field_order: tuple[str, ...]
    physical_extent: tuple[float, float, float, float]


def endpoint_indices(length: int, stride: int) -> np.ndarray:
    """Return exact integer samples with both physical endpoints present."""
    if isinstance(length, bool) or not isinstance(length, int) or length < _MINIMUM_AXIS_LENGTH:
        msg = "A display axis must have at least one coordinate."
        raise ValueError(msg)
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        msg = "Display stride must be a positive integer."
        raise ValueError(msg)
    indices = np.arange(0, length, stride, dtype=np.intp)
    if indices[-1] != length - 1:
        indices = np.append(indices, length - 1)
    indices.setflags(write=False)
    return indices


def resolve_display_stride(shape: tuple[int, int], selection: DisplayStride) -> int:
    """Select a deterministic map sampling rate without affecting reductions."""
    if selection not in (1, 2, 4, "auto") or isinstance(selection, bool):
        msg = "Display stride must be 1, 2, 4, or 'auto'."
        raise ValueError(msg)
    if len(shape) != _SPATIAL_AXES or min(shape) < _MINIMUM_AXIS_LENGTH:
        msg = "Display grid shape must contain two non-empty axes."
        raise ValueError(msg)
    if selection != "auto":
        return selection
    longest = max(shape)
    return 4 if longest > _AUTO_LARGE_AXIS else 2 if longest > _AUTO_MEDIUM_AXIS else 1


def display_indices(shape: tuple[int, int], selection: DisplayStride) -> tuple[np.ndarray, np.ndarray]:
    """Return presentation indices for the two trailing spatial axes."""
    stride = resolve_display_stride(shape, selection)
    return endpoint_indices(shape[0], stride), endpoint_indices(shape[1], stride)


def display_sample(values: np.ndarray, indices: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Sample aligned trailing spatial axes without modifying source values."""
    array = np.asanyarray(values)
    y_indices, x_indices = indices
    if array.ndim < _SPATIAL_AXES or array.shape[-2:] != (int(y_indices[-1]) + 1, int(x_indices[-1]) + 1):
        msg = "Display indices must end at the full array's spatial endpoints."
        raise ValueError(msg)
    return array[..., y_indices[:, None], x_indices]


def spatial_grid_identity(
    coordinates: np.ndarray,
    *,
    coordinate_units: tuple[str, str],
    field_order: tuple[str, ...],
    artifact_stride: int = 1,
    source_shape: tuple[int, int] | None = None,
    axis_indices: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> SpatialGridIdentity:
    """Bind exact coordinate bytes, source indices, extent, and field order."""
    array = np.asarray(coordinates)
    if (
        array.ndim != _COORDINATE_RANK
        or array.shape[0] != _SPATIAL_AXES
        or min(array.shape[-2:]) < _MINIMUM_AXIS_LENGTH
        or not np.issubdtype(array.dtype, np.floating)
    ):
        msg = "Coordinates must be floating x/y grids with shape [2,Y,X]."
        raise ValueError(msg)
    if not np.isfinite(array).all():
        msg = "Spatial coordinates must be finite."
        raise ValueError(msg)
    shape = (int(array.shape[1]), int(array.shape[2]))
    source = shape if source_shape is None else source_shape
    indices = (tuple(range(shape[0])), tuple(range(shape[1]))) if axis_indices is None else axis_indices
    if artifact_stride < 1 or len(indices[0]) != shape[0] or len(indices[1]) != shape[1]:
        msg = "Artifact index map does not match the coordinate shape."
        raise ValueError(msg)
    if any(
        values[0] != 0 or values[-1] != source[axis] - 1 or any(right <= left for left, right in pairwise(values))
        for axis, values in enumerate(indices)
    ):
        msg = "Artifact index map must be ordered and retain both endpoints."
        raise ValueError(msg)
    if len(coordinate_units) != _SPATIAL_AXES or not all(coordinate_units) or not field_order or len(set(field_order)) != len(field_order):
        msg = "Grid units and field order must be declared exactly."
        raise ValueError(msg)
    digest = hashlib.sha256()
    for component in array:
        contiguous = np.ascontiguousarray(component)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.tobytes())
    return SpatialGridIdentity(
        shape=shape,
        coordinate_digest=digest.hexdigest(),
        coordinate_units=coordinate_units,
        artifact_stride=artifact_stride,
        source_shape=source,
        axis_indices=indices,
        field_order=field_order,
        physical_extent=(float(np.min(array[0])), float(np.max(array[0])), float(np.min(array[1])), float(np.max(array[1]))),
    )


def require_common_grid(left: SpatialGridIdentity, right: SpatialGridIdentity) -> None:
    """Reject direct comparisons that differ in any spatial representation fact."""
    if left != right:
        changed = tuple(item.name for item in fields(SpatialGridIdentity) if getattr(left, item.name) != getattr(right, item.name))
        msg = (
            f"Incompatible Evaluation grids: {', '.join(changed)} differ. "
            f"Build each artifact on one explicit common grid with: {COMMON_GRID_ARTIFACT_COMMAND}"
        )
        raise ValueError(msg)
