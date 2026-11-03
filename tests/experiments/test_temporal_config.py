"""Explicit transient temporal configuration and identity contracts.

Responsibilities:
  - Check explicit temporal policies and sampling/conditioning identity changes

Design principles:
  - Missing policies fail rather than being inferred

This module does NOT:
  - Construct temporal tensors or execute rollout training
"""

from __future__ import annotations

import pytest

from src import datasets, learning
from src.experiments.config import experiments_config_loader as loader


def _temporal_config(*, sampling: dict[str, object], conditioning: str) -> dict[str, object]:
    """Resolve one complete test-owned temporal experiment branch."""
    return loader._resolve_temporal_config(
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
        loader._resolve_temporal_config({"sampling": {"mode": "one_step_transition"}})
    with pytest.raises(ValueError, match="rollout_length"):
        loader._resolve_temporal_config({"sampling": {"mode": "rollout_window"}, "temporal_conditioning": {"kind": "none"}})
    with pytest.raises(TypeError, match="explicit transient_sampling"):
        datasets.runtime.factory.DatasetRequest(
            dataset_id="synthetic_transient",
            dataset_view="transient_drying",
        )
    with pytest.raises(ValueError, match="cannot include transient_sampling"):
        datasets.runtime.factory.DatasetRequest(
            dataset_id="synthetic_steady",
            dataset_view="steady_flow",
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
