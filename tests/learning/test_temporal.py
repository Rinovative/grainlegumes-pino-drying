# ruff: noqa: S101
"""Explicit temporal-conditioning transformation contracts."""

from __future__ import annotations

import pytest
import torch

from src import learning


def test_temporal_conditioning_uses_configured_horizon_or_explicit_none() -> None:
    """Normalize by Generation's configured stop and preserve shape and dtype."""
    current_time = torch.tensor([0.0, 84.0, 168.0], dtype=torch.float32)
    normalized = learning.temporal.apply_temporal_conditioning(
        current_time,
        learning.temporal.TemporalConditioningSpec(kind="normalized_current_time"),
        configured_regular_horizon=168.0,
    )

    assert normalized is not None
    torch.testing.assert_close(normalized, torch.tensor([0.0, 0.5, 1.0]))
    assert normalized.shape == current_time.shape
    assert normalized.dtype == current_time.dtype
    assert (
        learning.temporal.apply_temporal_conditioning(
            current_time,
            learning.temporal.TemporalConditioningSpec(kind="none"),
            configured_regular_horizon=168.0,
        )
        is None
    )


def test_temporal_conditioning_rejects_inferred_or_incompatible_time_scales() -> None:
    """Reject absent policies, an early trajectory stop, and out-of-horizon time."""
    with pytest.raises(ValueError, match="keys must be exactly"):
        learning.temporal.TemporalConditioningSpec.from_mapping({})
    with pytest.raises(ValueError, match="Unknown temporal conditioning"):
        learning.temporal.TemporalConditioningSpec(kind="elapsed_fraction")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="within the configured horizon"):
        learning.temporal.apply_temporal_conditioning(
            torch.tensor([0.0, 84.0]),
            learning.temporal.TemporalConditioningSpec(kind="normalized_current_time"),
            configured_regular_horizon=2.5,
        )
