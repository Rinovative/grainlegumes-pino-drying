"""Resolve the endpoint-preserving strided view of the canonical grid that both tasks train on.

Responsibilities:
  - Validate an exact positive spatial stride without coercion
  - Refuse a stride that does not tile the canonical grid exactly, so both physical boundaries stay on it
  - Describe and digest the exact source-index map of one strided view

Design principles:
  - One resolution mechanism serves the steady and the transient Dataset runtimes, their splits and their artifacts
  - A view selects source indices; it never interpolates or changes stored values

This module does NOT:
  - Read Dataset payloads or apply a view to tensors
  - Choose the stride; ``data.spatial_stride`` of the run configuration does
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Final, cast

_SPATIAL_AXIS_COUNT: Final = 2
_MINIMUM_SPATIAL_AXIS_LENGTH: Final = 2


def validate_spatial_stride(value: Any) -> int:
    """Return one exact positive spatial stride without coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = "spatial_stride must be an integer."
        raise TypeError(message)
    if value < 1:
        message = "spatial_stride must be >= 1."
        raise ValueError(message)
    return value


def _validate_spatial_shape(value: Any, *, label: str) -> tuple[int, int]:
    """Return one exact two-axis spatial shape."""
    if (
        not isinstance(value, tuple)
        or len(value) != _SPATIAL_AXIS_COUNT
        or any(isinstance(axis, bool) or not isinstance(axis, int) or axis < _MINIMUM_SPATIAL_AXIS_LENGTH for axis in value)
    ):
        message = f"{label} must contain two integer spatial axes >= 2."
        raise ValueError(message)
    return cast("tuple[int, int]", value)


@dataclass(frozen=True, slots=True)
class SpatialRepresentation:
    """Describe one exact endpoint-preserving view of a canonical grid."""

    source_shape: tuple[int, int]
    spatial_stride: int
    represented_shape: tuple[int, int]
    y_indices: tuple[int, ...]
    x_indices: tuple[int, ...]
    _index_identity_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject shape, stride, endpoint, or index-map drift."""
        stride = validate_spatial_stride(self.spatial_stride)
        source_shape = _validate_spatial_shape(
            self.source_shape,
            label="source_shape",
        )
        expected_y = tuple(range(0, source_shape[0], stride))
        expected_x = tuple(range(0, source_shape[1], stride))
        represented_shape = (len(expected_y), len(expected_x))
        if represented_shape != self.represented_shape:
            message = "represented_shape does not match the source grid and spatial stride."
            raise ValueError(message)
        if self.y_indices != expected_y or self.x_indices != expected_x:
            message = "Spatial indices must be the exact endpoint-preserving strided source indices."
            raise ValueError(message)
        if expected_y[-1] != source_shape[0] - 1 or expected_x[-1] != source_shape[1] - 1:
            message = "Spatial indices must retain both physical endpoints."
            raise ValueError(message)
        object.__setattr__(self, "_index_identity_sha256", self._compute_index_identity_sha256())

    @property
    def index_identity_sha256(self) -> str:
        """Return the exact source-index-map identity."""
        return self._index_identity_sha256

    def _compute_index_identity_sha256(self) -> str:
        """Compute the immutable exact source-index-map identity once."""
        encoded = json.dumps(
            {
                "source_shape": list(self.source_shape),
                "spatial_stride": self.spatial_stride,
                "represented_shape": list(self.represented_shape),
                "y_indices": list(self.y_indices),
                "x_indices": list(self.x_indices),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Serialize the exact index-based spatial representation."""
        return {
            "source_shape": list(self.source_shape),
            "spatial_stride": self.spatial_stride,
            "represented_shape": list(self.represented_shape),
            "axis_indices": {
                "y": list(self.y_indices),
                "x": list(self.x_indices),
            },
            "index_identity_sha256": self.index_identity_sha256,
            "representation_transformation": "endpoint_preserving_strided_source_index_selection",
        }

    def view_evidence(self) -> dict[str, int]:
        """Return the canonical and effective view evidence that splits and scaling identities bind."""
        return {
            "spatial_stride": self.spatial_stride,
            "canonical_ny": self.source_shape[0],
            "canonical_nx": self.source_shape[1],
            "effective_ny": self.represented_shape[0],
            "effective_nx": self.represented_shape[1],
        }


def resolve_spatial_representation(
    canonical_shape: tuple[int, int],
    spatial_stride: int,
) -> SpatialRepresentation:
    """
    Return one exact endpoint-preserving canonical-grid representation.

    Raises
    ------
    ValueError
        If the shape is malformed or the stride does not tile the grid exactly, for example 4 on 251 x 401
        (250 is not divisible by 4), which would drop a physical boundary.

    """
    stride = validate_spatial_stride(spatial_stride)
    source_shape = _validate_spatial_shape(
        canonical_shape,
        label="canonical_shape",
    )
    if any((axis - 1) % stride != 0 for axis in source_shape):
        message = (
            f"spatial_stride={stride} does not tile the canonical {source_shape[0]} x {source_shape[1]} grid exactly: "
            f"{source_shape[0] - 1} and {source_shape[1] - 1} must both be divisible by it, so both physical boundaries stay on the grid."
        )
        raise ValueError(message)
    y_indices = tuple(range(0, source_shape[0], stride))
    x_indices = tuple(range(0, source_shape[1], stride))
    return SpatialRepresentation(
        source_shape=source_shape,
        spatial_stride=stride,
        represented_shape=(len(y_indices), len(x_indices)),
        y_indices=y_indices,
        x_indices=x_indices,
    )


def resolve_spatial_view(
    canonical_shape: tuple[int, int],
    spatial_stride: int,
) -> tuple[int, int]:
    """Return the boundary-preserving effective ``(Y, X)`` shape."""
    return resolve_spatial_representation(
        canonical_shape,
        spatial_stride,
    ).represented_shape
