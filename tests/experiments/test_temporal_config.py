# ruff: noqa: S101
"""Explicit transient temporal configuration and identity contracts."""

from __future__ import annotations

import pytest

from src import datasets, experiments, learning


def _temporal_config(*, sampling: dict[str, object], conditioning: str) -> dict[str, object]:
    """Resolve one complete test-owned temporal experiment branch."""
    return experiments.config.temporal.resolve_transient_temporal_config(
        {
            "sampling": sampling,
            "temporal_conditioning": {"kind": conditioning},
        }
    )


def test_temporal_config_requires_both_explicit_policies() -> None:
    """Reject missing modes and serialize exact one-step and rollout choices."""
    one_step = _temporal_config(
        sampling={"mode": "one_step_transition"},
        conditioning="normalized_current_time",
    )
    rollout = _temporal_config(
        sampling={
            "mode": "rollout_window",
            "rollout_length": 4,
            "window_stride": 2,
            "window_offset": 1,
        },
        conditioning="none",
    )

    assert one_step == {
        "sampling": {"mode": "one_step_transition"},
        "temporal_conditioning": {"kind": "normalized_current_time"},
    }
    assert rollout["sampling"] == {
        "mode": "rollout_window",
        "rollout_length": 4,
        "window_stride": 2,
        "window_offset": 1,
    }
    assert rollout["temporal_conditioning"] == {"kind": "none"}
    with pytest.raises(ValueError, match="keys must be exactly"):
        experiments.config.temporal.resolve_transient_temporal_config({"sampling": {"mode": "one_step_transition"}})
    with pytest.raises(ValueError, match="rollout_length"):
        experiments.config.temporal.resolve_transient_sampling({"mode": "rollout_window"})
    with pytest.raises(TypeError, match="explicit transient_sampling"):
        datasets.runtime.factory.DatasetRequest(
            dataset_id="synthetic_transient",
            dataset_view="transient_drying",
            evaluation_regime="id",
        )
    with pytest.raises(ValueError, match="cannot include transient_sampling"):
        datasets.runtime.factory.DatasetRequest(
            dataset_id="synthetic_steady",
            dataset_view="steady_flow",
            evaluation_regime="id",
            transient_sampling=datasets.contracts.transient.TransientSamplingSpec(mode="one_step_transition"),
        )


def test_sampling_and_conditioning_changes_update_run_and_resume_identity() -> None:
    """Bind temporal policy, window values, and conditioning mode to identities."""
    one_step = {
        "task": "transient_drying",
        "temporal": _temporal_config(
            sampling={"mode": "one_step_transition"},
            conditioning="normalized_current_time",
        ),
    }
    rollout = {
        "task": "transient_drying",
        "temporal": _temporal_config(
            sampling={
                "mode": "rollout_window",
                "rollout_length": 4,
                "window_stride": 1,
                "window_offset": 0,
            },
            conditioning="normalized_current_time",
        ),
    }
    no_conditioning = {
        "task": "transient_drying",
        "temporal": _temporal_config(
            sampling={"mode": "one_step_transition"},
            conditioning="none",
        ),
    }

    for changed in (rollout, no_conditioning):
        assert learning.training.checkpoint.config_digest(one_step) != learning.training.checkpoint.config_digest(changed)
        assert learning.training.checkpoint.resume_contract_digest(one_step) != learning.training.checkpoint.resume_contract_digest(changed)
