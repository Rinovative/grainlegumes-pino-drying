"""
analysis_field_maps.py

Define one exact discrete color-scale contract for scientific field maps.

Responsibilities:
  - Build truthful continuous, categorical, and constant field-map definitions
  - Preserve exact comparison boundaries and semantic colormap selection
  - Supply Matplotlib-ready normalization and rendering keyword arguments
  - Reuse identical in-process definitions through a bounded cache

Design principles:
  - Eleven visible colors require twelve strictly increasing continuous boundaries
  - Scientific values remain continuous and caller-owned arrays are never modified
  - Locked comparison scales preserve every boundary rather than only extrema

This module does NOT:
  - Render figures, select subplot layouts, or load scientific datasets
  - Derive scientific fields, units, or display transformations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Colormap, Normalize

from . import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from collections.abc import Sequence

N_COLOR_BINS: Final = 11
_CONTINUOUS_BOUNDARY_COUNT = N_COLOR_BINS + 1
_CACHE_SIZE = 128


class FieldMapSemantic(str, Enum):
    """Describe the scientific quantity semantics for a displayed field."""

    LINEAR_POSITIVE = "linear_positive"
    LINEAR_SIGNED = "linear_signed"
    LOG_POSITIVE = "log_positive"
    ABSOLUTE_ERROR = "absolute_error"
    SIGNED_ERROR = "signed_error"
    RELATIVE_ERROR = "relative_error"
    CATEGORICAL = "categorical"
    CONSTANT = "constant"


FieldMapKind = FieldMapSemantic
FieldMapExtend = Literal["neither", "min", "max", "both"]


@dataclass(frozen=True, slots=True)
class _ContinuousFieldMapState:
    """Hold cache-safe immutable state for one continuous visual scale."""

    semantic: FieldMapSemantic
    field: str
    unit: str
    display_transform: str
    comparison_scope: str | None
    bound_policy: str
    extend: FieldMapExtend
    locked: bool
    boundaries: tuple[float, ...]
    colormap_name: str


@dataclass(frozen=True, slots=True)
class FieldMapDefinition:
    """Hold one immutable scientific field-map scale and rendering state."""

    semantic: FieldMapSemantic
    field: str
    unit: str
    display_transform: str
    comparison_scope: str | None
    bound_policy: str
    extend: FieldMapExtend
    locked: bool
    boundaries: np.ndarray
    colormap: Colormap
    normalizer: Normalize
    ticks: np.ndarray
    category_labels: tuple[str, ...] = ()
    constant_value: float | None = None

    @property
    def color_count(self) -> int:
        """Return the number of visible colors in this definition."""
        return self.colormap.N

    @property
    def is_locked(self) -> bool:
        """Return whether this definition preserves supplied exact boundaries."""
        return self.locked

    def contourf_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments ready for a truthful ``Axes.contourf`` call."""
        kwargs = self._image_kwargs()
        if self.semantic is not FieldMapSemantic.CONSTANT:
            kwargs["levels"] = self.boundaries
            kwargs["extend"] = self.extend
        return kwargs

    def colorbar_kwargs(self) -> dict[str, Any]:
        """Return exact shared ticks and out-of-bound extension policy."""
        return {"ticks": self.ticks, "extend": self.extend}

    def pcolormesh_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments ready for an ``Axes.pcolormesh`` call."""
        return self._image_kwargs()

    def imshow_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments ready for an ``Axes.imshow`` call."""
        return self._image_kwargs()

    def encode_categories(self, values: Sequence[object] | np.ndarray) -> np.ndarray:
        """Return display codes for declared categories without mutating input."""
        if self.semantic is not FieldMapSemantic.CATEGORICAL:
            msg = "Category encoding is available only for categorical field maps."
            raise ValueError(msg)
        indices = {label: index for index, label in enumerate(self.category_labels)}
        array = np.asarray(values, dtype=object)
        try:
            return cast("np.ndarray", np.vectorize(lambda value: indices[str(value)], otypes=[np.int64])(array))
        except KeyError as error:
            msg = f"Categorical values include undeclared category {error.args[0]!r}."
            raise ValueError(msg) from error

    def _image_kwargs(self) -> dict[str, Any]:
        """Return common scalar-mappable arguments without caller data."""
        return {"cmap": self.colormap, "norm": self.normalizer}


def build_field_map(
    values: Sequence[object] | np.ndarray,
    *,
    semantic: FieldMapSemantic | str,
    field: str = "",
    unit: str = "",
    display_transform: str = "identity",
    comparison_scope: str | None = None,
    locked_boundaries: FieldMapDefinition | Sequence[float] | np.ndarray | None = None,
    bound_policy: str = "data",
    categories: Sequence[object] | None = None,
    colormap_name: str | None = None,
) -> FieldMapDefinition:
    """
    Build one exact semantic field-map definition without changing scientific values.

    Continuous definitions always contain exactly eleven colors and twelve strictly
    increasing boundaries. Categorical and constant definitions retain truthful
    cardinality instead of fabricating eleven intervals.
    """
    resolved = _semantic(semantic)
    metadata = _metadata(field, unit, display_transform, comparison_scope, bound_policy)
    if resolved is FieldMapSemantic.CATEGORICAL:
        return _categorical(values, metadata, categories, colormap_name)
    numeric = _numeric(values)
    _validate_values(numeric, resolved)
    is_constant = bool(np.all(numeric == numeric[0]))
    if resolved is FieldMapSemantic.CONSTANT and not is_constant:
        msg = "Constant field-map semantics require exactly one observed value."
        raise ValueError(msg)
    if resolved is FieldMapSemantic.CONSTANT or is_constant:
        if locked_boundaries is not None:
            msg = "Constant field maps do not accept fabricated locked interval boundaries."
            raise ValueError(msg)
        return _constant(numeric, metadata, colormap_name)
    boundaries, locked = _boundaries(numeric, resolved, locked_boundaries)
    boundaries = _validate_continuous_boundaries(boundaries)
    extend = _locked_extend(numeric, boundaries) if locked else "neither"
    state = _cached_continuous_state(
        resolved.value,
        *metadata,
        locked,
        extend,
        tuple(float(value) for value in boundaries),
        _colormap_name(resolved, field, colormap_name),
    )
    return _continuous_definition(state)


def field_map_cache_info() -> Any:
    """Return bounded in-process cache diagnostics for continuous scale state."""
    return _cached_continuous_state.cache_info()


def clear_field_map_cache() -> None:
    """Clear bounded in-process field-map and semantic-colormap reuse state."""
    _cached_continuous_state.cache_clear()
    _semantic_colormap.cache_clear()


def _semantic(value: FieldMapSemantic | str) -> FieldMapSemantic:
    """Normalize one declared field-map semantic."""
    if isinstance(value, FieldMapSemantic):
        return value
    if not isinstance(value, str):
        msg = "Field-map semantic must be FieldMapSemantic or text."
        raise TypeError(msg)
    try:
        return FieldMapSemantic(value)
    except ValueError as error:
        msg = f"Unsupported field-map semantic {value!r}."
        raise ValueError(msg) from error


def _metadata(field: str, unit: str, transform: str, scope: str | None, policy: str) -> tuple[str, str, str, str | None, str]:
    """Validate exact cache-relevant presentation metadata."""
    for label, value in (("field", field), ("unit", unit), ("display_transform", transform), ("bound_policy", policy)):
        if not isinstance(value, str):
            msg = f"{label} must be text."
            raise TypeError(msg)
        if value != value.strip():
            msg = f"{label} must not have surrounding whitespace."
            raise ValueError(msg)
    if scope is not None and (not isinstance(scope, str) or not scope or scope != scope.strip()):
        msg = "comparison_scope must be non-empty text without surrounding whitespace or None."
        raise ValueError(msg)
    return field, unit, transform, scope, policy


def _numeric(values: Sequence[object] | np.ndarray) -> np.ndarray:
    """Return finite numeric values without modifying caller-owned arrays."""
    array = np.asarray(values)
    if array.size == 0 or not np.issubdtype(array.dtype, np.number):
        msg = "Continuous field-map values must be one non-empty numeric array."
        raise ValueError(msg)
    numeric = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.isfinite(numeric).all():
        msg = "Continuous field-map values must be finite."
        raise ValueError(msg)
    return numeric


def _validate_values(values: np.ndarray, semantic: FieldMapSemantic) -> None:
    """Require values compatible with one declared numerical semantic."""
    if semantic is FieldMapSemantic.LOG_POSITIVE and np.any(values <= 0.0):
        msg = "Log-positive field maps require strictly positive values."
        raise ValueError(msg)
    if semantic in {FieldMapSemantic.LINEAR_POSITIVE, FieldMapSemantic.ABSOLUTE_ERROR, FieldMapSemantic.RELATIVE_ERROR} and np.any(values < 0.0):
        msg = f"{semantic.value} field maps require nonnegative values."
        raise ValueError(msg)


def _boundaries(
    values: np.ndarray, semantic: FieldMapSemantic, locked: FieldMapDefinition | Sequence[float] | np.ndarray | None
) -> tuple[np.ndarray, bool]:
    """Resolve exact locked boundaries or one semantic data-derived scale."""
    if locked is not None:
        return _locked(locked, semantic), True
    if semantic in {FieldMapSemantic.LINEAR_SIGNED, FieldMapSemantic.SIGNED_ERROR}:
        return np.linspace(-float(np.max(np.abs(values))), float(np.max(np.abs(values))), _CONTINUOUS_BOUNDARY_COUNT), False
    if semantic is FieldMapSemantic.LOG_POSITIVE:
        return np.geomspace(float(np.min(values)), float(np.max(values)), _CONTINUOUS_BOUNDARY_COUNT), False
    return np.linspace(float(np.min(values)), float(np.max(values)), _CONTINUOUS_BOUNDARY_COUNT), False


def _validate_continuous_boundaries(boundaries: np.ndarray) -> np.ndarray:
    """Require exactly twelve finite boundaries that remain distinct in float64."""
    if boundaries.shape != (_CONTINUOUS_BOUNDARY_COUNT,) or not np.isfinite(boundaries).all() or not np.all(np.diff(boundaries) > 0.0):
        msg = "Continuous field-map ranges must resolve to twelve finite strictly increasing boundaries."
        raise ValueError(msg)
    return boundaries


def _locked(locked: FieldMapDefinition | Sequence[float] | np.ndarray, semantic: FieldMapSemantic) -> np.ndarray:
    """Validate exact locked continuous boundaries without recomputation."""
    if isinstance(locked, FieldMapDefinition) and locked.semantic is not semantic:
        msg = "Locked field-map definitions must preserve the same scientific semantic."
        raise ValueError(msg)
    source = locked.boundaries if isinstance(locked, FieldMapDefinition) else locked
    boundaries = np.asarray(source, dtype=np.float64).reshape(-1)
    if boundaries.size != _CONTINUOUS_BOUNDARY_COUNT or not np.isfinite(boundaries).all() or not np.all(np.diff(boundaries) > 0.0):
        msg = "Locked continuous field-map boundaries must contain twelve finite strictly increasing values."
        raise ValueError(msg)
    if semantic in {FieldMapSemantic.LINEAR_SIGNED, FieldMapSemantic.SIGNED_ERROR} and (
        not np.isclose(boundaries[0], -boundaries[-1]) or not boundaries[5] < 0 < boundaries[6]
    ):
        msg = "Locked signed boundaries must be symmetric with a neutral center interval around zero."
        raise ValueError(msg)
    if semantic is FieldMapSemantic.LOG_POSITIVE and np.any(boundaries <= 0.0):
        msg = "Locked log-positive field-map boundaries must be strictly positive."
        raise ValueError(msg)
    return boundaries


def _locked_extend(values: np.ndarray, boundaries: np.ndarray) -> FieldMapExtend:
    """Expose locked-scale underflow or overflow without adding color intervals."""
    below = bool(np.any(values < boundaries[0]))
    above = bool(np.any(values > boundaries[-1]))
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def _constant(values: np.ndarray, metadata: tuple[str, str, str, str | None, str], override: str | None) -> FieldMapDefinition:
    """Build exact constant rendering state without manufacturing intervals."""
    field, unit, transform, scope, policy = metadata
    value = float(values[0])
    return FieldMapDefinition(
        FieldMapSemantic.CONSTANT,
        field,
        unit,
        transform,
        scope,
        policy,
        "neither",
        False,
        _readonly((value,)),
        _new_semantic_colormap(_colormap_name(FieldMapSemantic.CONSTANT, field, override), 1),
        Normalize(vmin=value, vmax=value, clip=True),
        _readonly((value,)),
        constant_value=value,
    )


def _categorical(
    values: Sequence[object] | np.ndarray, metadata: tuple[str, str, str, str | None, str], categories: Sequence[object] | None, override: str | None
) -> FieldMapDefinition:
    """Build actual-category state without artificial eleven-color expansion."""
    field, unit, transform, scope, policy = metadata
    source = tuple(categories) if categories is not None else tuple(np.asarray(values, dtype=object).reshape(-1))
    labels = tuple(dict.fromkeys(str(value) for value in source))
    if not labels or any(not label or label != label.strip() for label in labels):
        msg = "Categorical labels must be non-empty text without surrounding whitespace."
        raise ValueError(msg)
    if categories is not None and len(labels) != len(source):
        msg = "Declared categorical labels must be unique."
        raise ValueError(msg)
    count = len(labels)
    boundaries = _readonly(np.arange(count + 1, dtype=np.float64) - 0.5)
    cmap = _new_semantic_colormap(_colormap_name(FieldMapSemantic.CATEGORICAL, field, override), count)
    return FieldMapDefinition(
        FieldMapSemantic.CATEGORICAL,
        field,
        unit,
        transform,
        scope,
        policy,
        "neither",
        False,
        boundaries,
        cmap,
        BoundaryNorm(boundaries, cmap.N, clip=True),
        _readonly(np.arange(count, dtype=np.float64)),
        labels,
    )


@lru_cache(maxsize=_CACHE_SIZE)
def _cached_continuous_state(
    semantic_value: str,
    field: str,
    unit: str,
    transform: str,
    scope: str | None,
    policy: str,
    locked: bool,
    extend: FieldMapExtend,
    values: tuple[float, ...],
    cmap_name: str,
) -> _ContinuousFieldMapState:
    """Construct and retain only immutable exact continuous visual state."""
    return _ContinuousFieldMapState(
        FieldMapSemantic(semantic_value),
        field,
        unit,
        transform,
        scope,
        policy,
        extend,
        locked,
        values,
        cmap_name,
    )


def _continuous_definition(state: _ContinuousFieldMapState) -> FieldMapDefinition:
    """Create isolated mutable Matplotlib objects from cached immutable state."""
    boundaries = _readonly(state.boundaries)
    cmap = _new_semantic_colormap(state.colormap_name, N_COLOR_BINS)
    return FieldMapDefinition(
        state.semantic,
        state.field,
        state.unit,
        state.display_transform,
        state.comparison_scope,
        state.bound_policy,
        state.extend,
        state.locked,
        boundaries,
        cmap,
        BoundaryNorm(boundaries, cmap.N, clip=state.extend == "neither"),
        _readonly(boundaries),
    )


def _new_semantic_colormap(name: str, count: int) -> Colormap:
    """Return one caller-isolated copy of cached semantic colormap state."""
    return _semantic_colormap(name, count).copy()


@lru_cache(maxsize=32)
def _semantic_colormap(name: str, count: int) -> Colormap:
    """Return one private resampled template for an exact visible color count."""
    if count < 1:
        msg = "Semantic colormaps require at least one visible color."
        raise ValueError(msg)
    try:
        return colormaps[name].resampled(count)
    except KeyError as error:
        msg = f"Unknown Matplotlib colormap {name!r}."
        raise ValueError(msg) from error


def _colormap_name(semantic: FieldMapSemantic, field: str, override: str | None) -> str:
    """Resolve one semantic colormap while allowing an explicit override."""
    if override is not None:
        if not isinstance(override, str) or not override or override != override.strip():
            msg = "colormap_name must be non-empty text without surrounding whitespace or None."
            raise ValueError(msg)
        return override
    if semantic in {FieldMapSemantic.LINEAR_SIGNED, FieldMapSemantic.SIGNED_ERROR}:
        return visual_semantics.field_visual_semantics(field or "signed", role="signed_error").colormap
    if semantic is FieldMapSemantic.ABSOLUTE_ERROR:
        return visual_semantics.field_visual_semantics(field or "error", role="absolute_error").colormap
    if semantic is FieldMapSemantic.RELATIVE_ERROR:
        return "magma"
    if semantic is FieldMapSemantic.CATEGORICAL:
        return "tab20"
    return visual_semantics.field_visual_semantics(field or "scalar").colormap


def _readonly(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return one immutable bytes-backed float64 array."""
    source = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(source.tobytes(order="C"), dtype=np.float64).reshape(source.shape)
