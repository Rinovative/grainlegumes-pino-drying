"""Define one exact discrete color-scale contract for scientific field maps.

Responsibilities:
  - Build truthful continuous, categorical, and constant field-map definitions
  - Preserve exact comparison boundaries and semantic colormap selection
  - Supply Matplotlib-ready normalization and rendering keyword arguments
  - Give the robust percentile range of a state or an error, the one display scale rule of the Evaluation maps
  - Own the round band rules of every map: 9 to 13 equal bands of 1, 2, 2.5 or 5 times a power of ten, an odd count
    symmetric about 0 for a signed scale, and the error and state bands built on them
  - Own the power of ten of a state bar, from the size of its values (``state_exponent``): the state bands are built on
    the values divided by it, and the bar's labels are written in it

Design principles:
  - A data-derived linear scale has 9 to 13 round bands (signed: an odd count, the neutral band centred on 0), a
    log-positive one eleven geometric bands; a locked scale keeps its 9 to 13 bands exactly
  - Scientific values remain continuous and caller-owned arrays are never modified
  - Locked comparison scales preserve every boundary rather than only extrema
  - Every returned definition owns a fresh colormap and normalizer and read-only boundary arrays
  - A constant signed field keeps its signed colormap and the colour its sign has there: zero is the neutral centre

This module does NOT:
  - Render figures, select subplot layouts, write tick labels, or load scientific datasets
  - Derive scientific fields, units, or display transformations
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap, Normalize

from . import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

N_COLOR_BINS: Final = 11
MIN_BANDS: Final = 9  # a round scale has 9 to 13 bands (``rounded_boundaries``)
MAX_BANDS: Final = 13
DISPLAY_PERCENTILES: Final = (5.0, 95.0)  # owner 14.11.: a few extreme cells must not flatten a map
# An error bar's finer step moves its power of ten into the unit (owner 14.11.); a state bar's power of ten follows the size of its
# values, never its step, and its rules run on the values divided by it (``state_exponent``, owner 15.11.).
MAX_TICK_DECIMALS: Final = 3
_PLAIN_STATE_EXPONENT_LIMIT: Final = 4  # a state whose largest magnitude lies in [10^-3, 10^4) reads without a factor
_STATE_EXPONENT_PASSES: Final = 3  # banding again in a larger power of ten settles in a second pass; the third only guards
_CONTINUOUS_BOUNDARY_COUNT = N_COLOR_BINS + 1
_BAND_MANTISSAS = (1.0, 2.0, 2.5, 5.0)  # round band widths: these times a power of ten
_PREFERRED_BANDS = 11  # the band count that wins a tie in the covered span
_MAX_STEP_DECIMALS = 15
# Relative: far below the gap between round steps, above the roundoff of a step taken as a difference of large boundaries.
_STEP_TOLERANCE = 1e-6
_MIN_RECIPROCAL_EXPONENT = -300  # 10.0**300 is finite; the reciprocal of a smaller power of ten overflows
_MIN_STEP = 1e-305  # a smaller band step would need a decimal power below the normal doubles (10^-307), so no round bands


class FieldMapSemantic(str, Enum):  # noqa: UP042 -- string-enum str/repr behavior is an established presentation contract
    """Describe the scientific quantity semantics for a displayed field."""

    LINEAR_POSITIVE = "linear_positive"
    LINEAR_SEQUENTIAL = "linear_sequential"  # the state bands over the values' own range, of either sign (no zero centre, no positivity check)
    LINEAR_SIGNED = "linear_signed"
    LOG_POSITIVE = "log_positive"
    ABSOLUTE_ERROR = "absolute_error"
    SIGNED_ERROR = "signed_error"
    RELATIVE_ERROR = "relative_error"
    CATEGORICAL = "categorical"
    CONSTANT = "constant"


FieldMapKind = FieldMapSemantic
FieldMapExtend = Literal["neither", "min", "max", "both"]
SIGNED_SEMANTICS: Final = frozenset({FieldMapSemantic.LINEAR_SIGNED, FieldMapSemantic.SIGNED_ERROR})  # symmetric about 0, neutral centre band
ERROR_SEMANTICS: Final = frozenset({FieldMapSemantic.ABSOLUTE_ERROR, FieldMapSemantic.SIGNED_ERROR, FieldMapSemantic.RELATIVE_ERROR})


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

    Continuous definitions have one color per band: a data-derived linear state (positive, or sequential of either sign) the
    state bands over the values (``state_boundaries`` from their minimum to their maximum; signed: the same rules on
    ``symmetric_boundaries`` over their largest magnitude), a data-derived error the round bands (``rounded_boundaries``;
    signed: ``symmetric_boundaries``), a log-positive one eleven geometric bands, and locked boundaries their own 9 to 13
    bands. Categorical and constant definitions retain truthful cardinality instead of fabricating intervals.
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
        return _constant(numeric, metadata, colormap_name, signed=resolved in SIGNED_SEMANTICS)
    boundaries, locked = _boundaries(numeric, resolved, locked_boundaries)
    extend = locked_extend(numeric, boundaries) if locked else "neither"
    return _continuous(resolved, metadata, locked, extend, boundaries, _colormap_name(resolved, field, colormap_name))


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
    """Validate exact presentation metadata."""
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
    """
    Resolve exact locked boundaries or one semantic data-derived scale: geometric bands for a log-positive field, else round ones.

    A linear state takes the state bands (``state_boundaries``; signed: ``_signed_state_boundaries``), an error the plain round bands.
    """
    if locked is not None:
        return _locked(locked, semantic), True
    lower, upper, reach = float(np.min(values)), float(np.max(values)), float(np.max(np.abs(values)))
    found: np.ndarray | None
    if semantic is FieldMapSemantic.LOG_POSITIVE:
        found = np.geomspace(lower, upper, _CONTINUOUS_BOUNDARY_COUNT)
    elif semantic is FieldMapSemantic.LINEAR_SIGNED:
        found = _signed_state_boundaries(reach)
    elif semantic in {FieldMapSemantic.LINEAR_POSITIVE, FieldMapSemantic.LINEAR_SEQUENTIAL}:
        found = state_boundaries(lower, upper)
    else:
        found = symmetric_boundaries(reach) if semantic in SIGNED_SEMANTICS else rounded_boundaries(lower, upper)
    return _validate_continuous_boundaries(found), False


def _validate_continuous_boundaries(boundaries: np.ndarray | None) -> np.ndarray:
    """Require 9 to 13 bands on finite boundaries that remain distinct in float64 (``None``: no round scale exists)."""
    if (
        boundaries is None
        or not MIN_BANDS < boundaries.size <= MAX_BANDS + 1
        or not np.isfinite(boundaries).all()
        or not np.all(np.diff(boundaries) > 0.0)
    ):
        msg = f"Continuous field-map ranges must resolve to {MIN_BANDS + 1} to {MAX_BANDS + 1} finite strictly increasing boundaries."
        raise ValueError(msg)
    return boundaries


def _locked(locked: FieldMapDefinition | Sequence[float] | np.ndarray, semantic: FieldMapSemantic) -> np.ndarray:
    """Validate exact locked continuous boundaries without recomputation."""
    if isinstance(locked, FieldMapDefinition) and locked.semantic is not semantic:
        msg = "Locked field-map definitions must preserve the same scientific semantic."
        raise ValueError(msg)
    source = locked.boundaries if isinstance(locked, FieldMapDefinition) else locked
    boundaries = np.asarray(source, dtype=np.float64).reshape(-1)
    if not MIN_BANDS < boundaries.size <= MAX_BANDS + 1 or not np.isfinite(boundaries).all() or not np.all(np.diff(boundaries) > 0.0):
        msg = f"Locked continuous field-map boundaries must contain {MIN_BANDS + 1} to {MAX_BANDS + 1} finite strictly increasing values."
        raise ValueError(msg)
    centre = (boundaries.size - 1) // 2  # the first boundary above the neutral band of an odd band count
    if semantic in SIGNED_SEMANTICS and (
        not np.isclose(boundaries[0], -boundaries[-1]) or boundaries.size % 2 != 0 or not boundaries[centre] < 0 < boundaries[centre + 1]
    ):
        msg = "Locked signed boundaries must be symmetric with a neutral center interval around zero."
        raise ValueError(msg)
    if semantic is FieldMapSemantic.LOG_POSITIVE and np.any(boundaries <= 0.0):
        msg = "Locked log-positive field-map boundaries must be strictly positive."
        raise ValueError(msg)
    return boundaries


def percentile_scale(values: np.ndarray, kind: Literal["state", "signed", "magnitude"]) -> tuple[float, float] | None:
    """
    Return the robust display range of the finite ``values`` (``None`` without any); values beyond it get the bar's arrows.

    A ``state`` runs from the 5th to the 95th percentile, a ``signed`` error symmetric to the larger absolute one of the
    two, a ``magnitude`` (absolute or normalized error) from 0 to the 95th percentile. The range may be degenerate.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    lower, upper = (float(value) for value in np.percentile(finite, DISPLAY_PERCENTILES))
    if kind == "signed":
        reach = max(abs(lower), abs(upper))
        return -reach, reach
    return (lower, upper) if kind == "state" else (0.0, upper)


def locked_extend(values: np.ndarray, boundaries: np.ndarray) -> FieldMapExtend:
    """Return the colour-bar arrows for values below the first or above the last boundary, without adding colour intervals."""
    below = bool(np.any(values < boundaries[0]))
    above = bool(np.any(values > boundaries[-1]))
    if below and above:
        return "both"
    if below:
        return "min"
    if above:
        return "max"
    return "neither"


def step_decimals(step: float) -> int:
    """Return the decimals the round ``step`` needs exactly (0.25 gives 2, 2.5 gives 1, 25 gives 0)."""
    for decimals in range(_MAX_STEP_DECIMALS):
        if abs(round(step, decimals) - step) <= _STEP_TOLERANCE * step:
            return decimals
    return _MAX_STEP_DECIMALS


def _best_band(width: float, bands_from: Callable[[float], int | None], max_decimals: int | None) -> tuple[float, int] | None:
    """Return the (step, band count) of the round candidate covering the least span, ties to the count nearest 11, then the larger step."""
    base = math.floor(math.log10(width / MAX_BANDS))
    best: tuple[tuple[float, int, float], float, int] | None = None
    for power in range(base, base + 3):
        for mantissa in _BAND_MANTISSAS:
            step = mantissa * 10.0**power
            count = None if max_decimals is not None and step_decimals(step) > max_decimals else bands_from(step)
            if count is None:
                continue
            key = (round(count * step / width, 9), abs(count - _PREFERRED_BANDS), -step)
            if best is None or key < best[0]:
                best = (key, step, count)
    return None if best is None else best[1:]


def rounded_boundaries(lower: float, upper: float, *, max_decimals: int | None = None) -> np.ndarray | None:
    """
    Return the round boundaries of 9 to 13 equal bands from a multiple of the step at or below ``lower`` that reach ``upper``.

    The step is m times a power of ten with m in 1, 2, 2.5 or 5. Among the candidates that reach ``upper`` the one with the
    smallest covered span wins; a tie goes to the band count nearest 11, then to the larger step. With ``max_decimals``
    candidates whose step needs more decimals are skipped, and a range whose best step is finer than the last decimal is too
    narrow for it: ``None``, as for ``upper`` not above ``lower``, a width that overflows, or one whose step would fall
    below ``_MIN_STEP``. The boundaries are the doubles nearest their decimal values (``_multiples``).
    """
    width = upper - lower
    if not (math.isfinite(width) and width / MAX_BANDS >= _MIN_STEP):
        return None

    def bands_from(step: float) -> int | None:
        first = _first_multiple(lower, step)
        count = max(MIN_BANDS, math.ceil(upper / step - first - 1e-9))
        count += 1 if _multiples(first + count, step) < upper else 0  # the top itself is checked: a rounding step below ``upper`` would cut it short
        return count if count <= MAX_BANDS else None

    found = _capped_band(width, bands_from, max_decimals)
    if found is None:
        return None
    step, count = found
    return _multiples(_first_multiple(lower, step) + np.arange(count + 1), step)


def _capped_band(width: float, bands_from: Callable[[float], int | None], max_decimals: int | None) -> tuple[float, int] | None:
    """Return ``_best_band``; with ``max_decimals`` one needing more decimals gives way to the best that does not, ``None`` if finer than the last."""
    found = _best_band(width, bands_from, None)
    if found is not None and max_decimals is not None and step_decimals(found[0]) > max_decimals:
        found = None if found[0] < 10.0**-max_decimals * (1.0 - 1e-9) else _best_band(width, bands_from, max_decimals)
    return found


def _multiples(indices: np.ndarray | int, step: float) -> np.ndarray:
    """Return ``indices`` times the round ``step`` as the doubles nearest their decimal values (3 times 0.1 gives 0.3)."""
    exponent = math.floor(math.log10(step)) - 1  # m times 10^k with m in 1, 2, 2.5 or 5 is a whole number times 10^(k-1)
    whole = np.asarray(indices, dtype=np.float64) * round(step / 10.0**exponent)
    # Dividing by 10^-k keeps 0.3 exact; below 10^-300 that reciprocal overflows and only the product is representable.
    return whole / 10.0**-exponent if _MIN_RECIPROCAL_EXPONENT < exponent < 0 else whole * 10.0**exponent


def _first_multiple(lower: float, step: float) -> int:
    """Return the index of the largest multiple of ``step`` at or below ``lower`` (a quotient rounded up to a whole number steps back)."""
    first = math.floor(lower / step + 1e-9)
    return first - 1 if _multiples(first, step) > lower else first


def symmetric_boundaries(reach: float, *, max_decimals: int | None = None) -> np.ndarray | None:
    """
    Return round boundaries of an odd number (9, 11 or 13) of equal bands of width w with ``count / 2 * w`` at least ``reach``.

    The middle band is centred on 0. The candidates, the choice among them and ``max_decimals`` are those of
    ``rounded_boundaries``. ``None`` for a reach whose step would fall below ``_MIN_STEP`` (or that is not positive), whose
    span overflows, or whose best step is finer than the last of ``max_decimals``.
    """
    if not (math.isfinite(2.0 * reach) and reach / MAX_BANDS >= _MIN_STEP):
        return None

    def bands_from(step: float) -> int | None:
        count = max(MIN_BANDS, math.ceil(2.0 * reach / step - 1e-9))
        count += 1 - count % 2
        count += 2 if count / 2.0 * step < reach else 0
        return count if count <= MAX_BANDS else None

    found = _capped_band(2.0 * reach, bands_from, max_decimals)
    if found is None:
        return None
    step, count = found
    return (np.arange(count + 1) - count / 2.0) * step


def error_boundaries(values: np.ndarray, *, signed: bool) -> np.ndarray:
    """
    Return the round error bands of the finite error ``values``: from 0 to the round top over their reach, or symmetric about 0 if signed.

    The reach is the 95th percentile (signed: the larger absolute one of the 5th and 95th). The bands are those of
    ``rounded_boundaries``; a signed scale has an odd count with the neutral band centred on 0 (``symmetric_boundaries``).
    A map without any error (reach 0) gets the scale of reach 1, so it never degenerates to one colour.
    """
    scale = percentile_scale(values, "signed" if signed else "magnitude")
    if scale is None:
        msg = "No round error bands exist for values without a finite error."
        raise ValueError(msg)
    reach = scale[1] if scale[1] > 0.0 else 1.0
    boundaries = symmetric_boundaries(reach) if signed else rounded_boundaries(0.0, reach)
    if boundaries is None:
        msg = f"No round error bands exist for a reach of {reach!r}."
        raise ValueError(msg)
    # BoundaryNorm sends a value equal to its last boundary to the over colour; an error exactly at the round top belongs to the scale.
    boundaries[-1] = np.nextafter(boundaries[-1], np.inf)
    return boundaries


def state_exponent(lower: float, upper: float) -> int:
    """
    Return the power of ten a state bar from ``lower`` to ``upper`` is written in: it follows the size of the values, never the step.

    With M the larger of the two magnitudes it is floor(log10 M), but 0 for M = 0 or 10^-3 <= M < 10^4, where the
    temperature, the porosity and the moistures lie. ``state_boundaries`` bands in this unit and the bar labels in it.
    """
    reach = max(abs(lower), abs(upper))
    # The tolerance keeps a boundary one rounding below a power of ten (10 times 1e-11) in that power.
    exponent = math.floor(math.log10(reach) + 1e-9) if reach > 0.0 else 0
    return 0 if -MAX_TICK_DECIMALS <= exponent < _PLAIN_STATE_EXPONENT_LIMIT else exponent


def state_boundaries(lower: float, upper: float) -> np.ndarray:
    """
    Return the round state boundaries (9 to 13 bands) from ``lower`` that reach ``upper``, never cut short, in physical units.

    Every rule runs on the values divided by 10**``state_exponent``: a constant value spans its magnitude (at least 1)
    upward, so it fills the lowest band of a readable bar, and a near-uniform range whose round step is finer than the
    three decimals a state bar may show is widened to 11 bands of 0.001 about its centre. The exponent is the returned
    bar's own, the one its labels take from its boundaries: bands that round up across a power of ten are built again in it.
    """

    def bands(low: float, top: float) -> np.ndarray:
        top = top if upper > lower else low + max(abs(low), 1.0)
        found = rounded_boundaries(low, top, max_decimals=MAX_TICK_DECIMALS)
        return np.linspace(*widened_range(low, top, MAX_TICK_DECIMALS), N_COLOR_BINS + 1) if found is None else found

    return _settled_state_bands(lower, upper, bands)


def _signed_state_boundaries(reach: float) -> np.ndarray:
    """
    Return the signed state boundaries symmetric about 0 that reach ``reach``, by the rules of ``state_boundaries``, in physical units.

    The bands are those of ``symmetric_boundaries`` on the values divided by 10**``state_exponent``, with at most three
    decimals there; a reach whose round step would be finer gets the 11 bands of 0.001 centred on 0, the finest the bar shows.
    """

    def bands(_: float, top: float) -> np.ndarray:
        found = symmetric_boundaries(top, max_decimals=MAX_TICK_DECIMALS)
        return (np.arange(N_COLOR_BINS + 1) - N_COLOR_BINS / 2.0) * 10.0**-MAX_TICK_DECIMALS if found is None else found

    return _settled_state_bands(-reach, reach, bands)


def _settled_state_bands(lower: float, upper: float, bands: Callable[[float, float], np.ndarray]) -> np.ndarray:
    """Return 10**e times the ``bands`` of ``lower`` and ``upper`` divided by it, with e the ``state_exponent`` the result settles in."""
    exponent = state_exponent(lower, upper)
    for _ in range(_STATE_EXPONENT_PASSES):
        scale = 10.0**exponent
        boundaries = scale * bands(lower / scale, upper / scale)
        settled = state_exponent(float(boundaries[0]), float(boundaries[-1]))
        if settled == exponent:
            return boundaries
        exponent = settled
    msg = f"The state bands of {lower!r} to {upper!r} never settle on one power of ten."
    raise RuntimeError(msg)


def widened_range(lower: float, upper: float, decimals: int) -> tuple[float, float]:
    """
    Return a range widened symmetrically about its centre to 11 steps of 10**-``decimals``, starting on a whole step.

    A nonnegative range (``lower`` at least 0) never reaches below 0: a window that would is shifted to start at 0.
    """
    unit = 10.0**-decimals
    start = round(((lower + upper) / 2.0 - 5.5 * unit) / unit) * unit
    start = max(start, 0.0) if lower >= 0.0 else start
    return start, start + 11.0 * unit


def _constant(values: np.ndarray, metadata: tuple[str, str, str, str | None, str], override: str | None, *, signed: bool) -> FieldMapDefinition:
    """
    Build exact constant rendering state without manufacturing intervals.

    A ``signed`` constant keeps its signed colormap and takes the one colour its value has on a symmetric scale reaching it: the
    neutral centre for zero, the low end for a negative and the high end for a positive value.
    """
    field, unit, transform, scope, policy = metadata
    value = float(values[0])
    if signed:
        name = _colormap_name(FieldMapSemantic.LINEAR_SIGNED, field, override)
        ends_and_centre = _new_semantic_colormap(name, 3)  # the low end, the neutral centre and the high end
        colormap: Colormap = ListedColormap(np.asarray([ends_and_centre(1 + int(np.sign(value)))]), name=name)
    else:
        colormap = _new_semantic_colormap(_colormap_name(FieldMapSemantic.CONSTANT, field, override), 1)
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
        colormap,
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


def _continuous(
    semantic: FieldMapSemantic,
    metadata: tuple[str, str, str, str | None, str],
    locked: bool,
    extend: FieldMapExtend,
    values: np.ndarray,
    cmap_name: str,
) -> FieldMapDefinition:
    """Build continuous rendering state with read-only boundaries and fresh Matplotlib objects."""
    field, unit, transform, scope, policy = metadata
    boundaries = _readonly(values)
    cmap = _new_semantic_colormap(cmap_name, boundaries.size - 1)
    return FieldMapDefinition(
        semantic,
        field,
        unit,
        transform,
        scope,
        policy,
        extend,
        locked,
        boundaries,
        cmap,
        BoundaryNorm(boundaries, cmap.N, clip=extend == "neither"),
        _readonly(boundaries),
    )


def _new_semantic_colormap(name: str, count: int) -> Colormap:
    """Return one freshly resampled colormap for an exact visible color count."""
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
    if semantic in SIGNED_SEMANTICS:
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
