"""
learning_temporal.py

Validate and apply explicit temporal conditioning for transient learning.
Responsibilities:
  - Declare supported temporal-conditioning identities
  - Admit exact conditioning mappings without hidden defaults
  - Normalize current regular time by the Dataset-configured horizon
Design principles:
  - Dataset runtime owns physical time and the configured scientific horizon
  - Learning owns only the model-facing conditioning transformation
  - Disabled conditioning returns no feature rather than a fabricated constant
This module does NOT:
  - Select transient samples, read HDF5 data, or infer a time scale
  - Concatenate temporal features into any particular model architecture
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal, TypeAlias

import torch

TemporalConditioningKind: TypeAlias = Literal["normalized_current_time", "none"]
TEMPORAL_CONDITIONING_KINDS: Final[tuple[TemporalConditioningKind, ...]] = (
    "normalized_current_time",
    "none",
)


@dataclass(frozen=True, slots=True)
class TemporalConditioningSpec:
    """Declare one explicit model-facing temporal conditioning policy."""

    kind: TemporalConditioningKind

    def __post_init__(self) -> None:
        """Reject unknown temporal-conditioning identities."""
        if self.kind not in TEMPORAL_CONDITIONING_KINDS:
            available = ", ".join(TEMPORAL_CONDITIONING_KINDS)
            message = f"Unknown temporal conditioning kind {self.kind!r}. Available kinds: {available}."
            raise ValueError(message)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TemporalConditioningSpec:
        """Resolve one exact conditioning mapping without a default kind."""
        if not isinstance(value, Mapping):
            message = "Temporal conditioning configuration must be a mapping."
            raise TypeError(message)
        if set(value) != {"kind"}:
            message = "Temporal conditioning keys must be exactly ['kind']."
            raise ValueError(message)
        return cls(kind=value["kind"])

    def as_dict(self) -> dict[str, str]:
        """Serialize the exact conditioning identity."""
        return {"kind": self.kind}


def apply_temporal_conditioning(
    t_n: torch.Tensor,
    spec: TemporalConditioningSpec,
    *,
    configured_regular_horizon: float,
) -> torch.Tensor | None:
    """
    Return normalized current time or no feature for an explicit policy.

    Parameters
    ----------
    t_n : torch.Tensor
        Scalar or rollout-step vector read from the canonical HDF5 regular-time
        axis by the Dataset runtime.
    spec : TemporalConditioningSpec
        Explicit conditioning identity.
    configured_regular_horizon : float
        Positive ``generation_scientific_config.time.stop`` admitted by Dataset.

    Returns
    -------
    torch.Tensor or None
        A same-shape dimensionless tensor for ``normalized_current_time`` or
        ``None`` when conditioning is explicitly disabled.

    """
    if not isinstance(spec, TemporalConditioningSpec):
        message = "spec must be one validated TemporalConditioningSpec."
        raise TypeError(message)
    if spec.kind == "none":
        return None
    if not isinstance(t_n, torch.Tensor) or not torch.is_floating_point(t_n) or t_n.numel() == 0:
        message = "Normalized current time requires one non-empty floating-point tensor."
        raise TypeError(message)
    if (
        isinstance(configured_regular_horizon, bool)
        or not isinstance(configured_regular_horizon, Real)
        or not math.isfinite(float(configured_regular_horizon))
        or float(configured_regular_horizon) <= 0.0
    ):
        message = "configured_regular_horizon must be one positive finite value."
        raise ValueError(message)
    horizon = t_n.new_tensor(float(configured_regular_horizon))
    if not bool(torch.isfinite(t_n).all()) or bool((t_n < 0.0).any()) or bool((t_n > horizon).any()):
        message = "Current regular time must be finite and within the configured horizon."
        raise ValueError(message)
    normalized = t_n / horizon
    if normalized.shape != t_n.shape:
        message = "Temporal conditioning must preserve the input time shape."
        raise RuntimeError(message)
    return normalized
