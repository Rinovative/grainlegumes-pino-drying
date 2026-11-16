"""Protect the one Train scaling format that both tasks fit, persist and apply.

Responsibilities:
  - Check the shared scaling class and persisted format and the refusal of old payloads
  - Check the steady nondimensionalization by the mean inlet pressure: fit, encoding, round trip, equivariance and refusals

Design principles:
  - Fixtures are tiny test-owned grids with known statistics

This module does NOT:
  - Test the transient case-ordered fit, which the transient foundation tests own, or the scaling cache
"""

from __future__ import annotations

import pytest
import torch
from support.scaling import spatial_view, steady_inputs, transient_scaling

from src import datasets, domain

_SHAPE = (3, 4)
_TASK = domain.tasks.registry.get_task("steady_flow")
_P_IN_BC = _TASK.input_names.index("p_in_bc")


def _steady_items() -> list[dict[str, torch.Tensor]]:
    """Return two steady Train cases on a coordinate grid with distinct inlet pressures, static fields and outputs."""
    base = torch.linspace(0.0, 1.0, _SHAPE[0] * _SHAPE[1]).reshape(*_SHAPE)
    items = []
    for case, inlet_pressure in enumerate((50.0, 400.0)):
        inputs = steady_inputs(1, _SHAPE, inlet_pressure=inlet_pressure)[0]
        for index, name in enumerate(("Kxx", "Kxy", "Kyy", "eps_bed"), start=1):
            inputs[_TASK.input_names.index(name)] = base * index + case
        outputs = torch.stack([inlet_pressure * (base * (index + 1) + 0.5 * case) for index in range(_TASK.out_channels)])
        items.append({"x": inputs, "y": outputs})
    return items


def _steady_scaling(items: list[dict[str, torch.Tensor]] | None = None) -> datasets.preprocessing.normalization.TrainScaling:
    """Fit the steady scaling on the given (by default the two) test cases."""
    normalization = datasets.preprocessing.normalization
    return normalization.fit_steady_scaling(
        _steady_items() if items is None else items,
        preprocessing=normalization.preprocessing_spec("steady_flow", spatial_view=spatial_view(_SHAPE)),
        dataset_identity={"dataset_id": "synthetic"},
        train_membership_digest="b" * 64,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two test cases as one physical input and output batch."""
    items = _steady_items()
    return torch.stack([item["x"] for item in items]), torch.stack([item["y"] for item in items])


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


def test_steady_fit_standardizes_the_dimensionless_states_and_keeps_the_physical_ones() -> None:
    """The state group holds the case-equal moments of y/P, the physical state group those of y."""
    scaling = _steady_scaling()
    inputs, outputs = _batch()
    reference = scaling.state_reference(inputs)
    torch.testing.assert_close(reference, torch.tensor([50.0, 400.0]))
    for group, values in (("state", outputs / reference.view(-1, 1, 1, 1)), ("physical_state", outputs)):
        variance, mean = torch.var_mean(values.double(), dim=(0, 2, 3), correction=0)
        torch.testing.assert_close(scaling.group(group).mean, mean.float())
        torch.testing.assert_close(scaling.group(group).std, variance.sqrt().float())
    assert scaling.physical_state is scaling.group("physical_state")


def test_steady_encoding_divides_by_the_reference_and_decodes_exactly_back() -> None:
    """p_in_bc/P takes the statistics of p/P, static inputs their own; decoding with P recovers the physical values."""
    scaling = _steady_scaling()
    inputs, outputs = _batch()
    reference = scaling.state_reference(inputs)
    encoded = scaling.encode_inputs(inputs, reference=reference)
    state, static = scaling.group("state"), scaling.group("static")
    pressure = state.names.index("p")
    expected_boundary = (inputs[:, _P_IN_BC] / reference.view(-1, 1, 1) - state.mean[pressure]) / state.scale[pressure]
    torch.testing.assert_close(encoded[:, _P_IN_BC], expected_boundary)
    for name in static.names:
        channel, index = _TASK.input_names.index(name), static.names.index(name)
        torch.testing.assert_close(encoded[:, channel], (inputs[:, channel] - static.mean[index]) / static.scale[index])
    torch.testing.assert_close(scaling.decode_inputs(encoded, reference=reference), inputs)
    torch.testing.assert_close(scaling.decode_state(scaling.encode_state(outputs, reference=reference), reference=reference), outputs)


def test_steady_encoding_is_invariant_when_a_case_is_scaled_with_its_inlet_pressure() -> None:
    """Multiplying p_in_bc and y of a case by c > 0 leaves its encoded inputs and targets unchanged; c = 4 keeps it bit-exact."""
    scaling = _steady_scaling()
    inputs, outputs = _batch()
    scaled_inputs = inputs.clone()
    scaled_inputs[:, _P_IN_BC] *= 4.0
    reference, scaled_reference = scaling.state_reference(inputs), scaling.state_reference(scaled_inputs)
    assert torch.equal(scaling.encode_inputs(scaled_inputs, reference=scaled_reference), scaling.encode_inputs(inputs, reference=reference))
    assert torch.equal(scaling.encode_state(4.0 * outputs, reference=scaled_reference), scaling.encode_state(outputs, reference=reference))


def test_the_steady_encoding_needs_the_reference_and_the_transient_one_refuses_it() -> None:
    """A steady encode or decode without P and a transient one with P fail instead of silently using another representation."""
    steady = _steady_scaling()
    inputs, outputs = _batch()
    with pytest.raises(ValueError, match="requires the per-case reference"):
        steady.encode_state(outputs)
    with pytest.raises(ValueError, match="requires the per-case reference"):
        steady.decode_inputs(inputs)
    transient = transient_scaling(spatial_view(_SHAPE))
    with pytest.raises(ValueError, match="takes no per-case reference"):
        transient.encode_state(torch.zeros((1, 4, *_SHAPE)), reference=torch.ones(1))
    with pytest.raises(ValueError, match="Only the steady representation"):
        transient.state_reference(inputs)


def test_a_non_positive_mean_inlet_pressure_is_refused_naming_the_case() -> None:
    """The fit names the Train case whose mean prescribed inlet pressure cannot divide the states."""
    items = _steady_items()
    items[1]["x"][_P_IN_BC] = 0.0
    with pytest.raises(ValueError, match=r"Steady case 1 has mean prescribed inlet pressure 0\.0"):
        _steady_scaling(items)


def test_a_steady_state_without_the_inlet_pressure_reference_is_refused() -> None:
    """A steady state saved before the nondimensionalization (no state reference, no physical group) is not read."""
    state = _steady_scaling().state_dict()
    del state["preprocessing"]["state_reference"]
    del state["groups"]["physical_state"]
    with pytest.raises(ValueError, match="must name the state reference 'mean_inlet_pressure'"):
        datasets.preprocessing.normalization.TrainScaling.from_state_dict(state)


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
