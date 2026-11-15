"""Protect the one Train scaling format that both tasks fit, persist and apply.

Responsibilities:
  - Check the shared scaling class and persisted format, the steady boundary-input scale and the refusal of old payloads

Design principles:
  - Fixtures are tiny test-owned grids with known statistics

This module does NOT:
  - Test the transient case-ordered fit, which the transient foundation tests own, or the scaling cache
"""

from __future__ import annotations

import pytest
import torch
from support.scaling import spatial_view, transient_scaling

from src import datasets, domain

_SHAPE = (3, 4)


def _steady_items() -> list[dict[str, torch.Tensor]]:
    """Return two steady Train cases whose channels carry distinct constant offsets."""
    task = domain.tasks.registry.get_task("steady_flow")
    base = torch.linspace(0.0, 1.0, _SHAPE[0] * _SHAPE[1]).reshape(1, *_SHAPE)
    return [
        {
            "x": torch.cat([base * (index + 1) + case for index in range(task.in_channels)]),
            "y": torch.cat([base * 10.0 * (index + 1) + 5.0 * case for index in range(task.out_channels)]),
        }
        for case in range(2)
    ]


def _steady_scaling() -> datasets.preprocessing.normalization.TrainScaling:
    """Fit the steady scaling on the two test cases."""
    normalization = datasets.preprocessing.normalization
    return normalization.fit_steady_scaling(
        _steady_items(),
        preprocessing=normalization.preprocessing_spec("steady_flow", spatial_view=spatial_view(_SHAPE)),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest="b" * 64,
    )


def test_both_tasks_persist_one_scaling_class_and_format() -> None:
    """A steady fit and a transient scaling are the same class, share one state schema and round-trip unchanged."""
    normalization = datasets.preprocessing.normalization
    steady = _steady_scaling()
    transient = transient_scaling(spatial_view(_SHAPE))
    assert type(steady) is type(transient) is normalization.TrainScaling
    steady_state, transient_state = steady.state_dict(), transient.state_dict()
    assert steady_state.keys() == transient_state.keys()
    assert steady_state["schema_kind"] == transient_state["schema_kind"] == normalization.SCALING_SCHEMA_KIND
    for scaling, state in ((steady, steady_state), (transient, transient_state)):
        assert normalization.TrainScaling.from_state_dict(state).digest == scaling.digest


def test_steady_inlet_pressure_uses_the_mean_and_scale_of_p() -> None:
    """p_in_bc is encoded with the Train statistics of the pressure state, every static input with its own."""
    scaling = _steady_scaling()
    task = domain.tasks.registry.get_task("steady_flow")
    inputs = torch.stack([item["x"] for item in _steady_items()])
    encoded = scaling.encode_inputs(inputs)
    state, static = scaling.group("state"), scaling.group("static")
    boundary = task.input_names.index("p_in_bc")
    pressure = state.names.index("p")
    expected_boundary = (inputs[:, boundary] - state.mean[pressure]) / state.scale[pressure]
    torch.testing.assert_close(encoded[:, boundary], expected_boundary)
    for name in static.names:
        channel, index = task.input_names.index(name), static.names.index(name)
        torch.testing.assert_close(encoded[:, channel], (inputs[:, channel] - static.mean[index]) / static.scale[index])
    torch.testing.assert_close(scaling.decode_inputs(encoded), inputs)


@pytest.mark.parametrize(
    "state",
    [
        {"in_normalizer.mean": torch.zeros(1, 7, 1, 1), "out_normalizer.std": torch.ones(1, 3, 1, 1)},
        {"schema_kind": "transient_train_scaling", "scale_mode": "state_std", "state_std": [1.0, 1.0, 1.0, 1.0]},
    ],
    ids=("neuralop-processor", "transient-group-scaling"),
)
def test_normalizers_fitted_before_the_common_scaling_are_refused(state: dict[str, object]) -> None:
    """Old steady and transient normalizer payloads fail with one clear message instead of being read."""
    with pytest.raises(ValueError, match="not the current 'train_channel_scaling' format"):
        datasets.preprocessing.normalization.TrainScaling.from_state_dict(state)
