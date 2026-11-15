"""Protect semantic supervised/physics loss composition and deterministic warmup.

Responsibilities:
  - Check named per-case supervised/physics components, warmup, normalization, telemetry, and monitor
    failures
  - Check the group-weighted state loss of both tasks, its velocity vector term, and its absolute H1
    option in metres
  - Check that every physics term carries gradients through the normalizers to the prediction
  - Check that a steady PI update gives the same step for any memory split of its cases

Design principles:
  - Disabled physics has explicit zero components

This module does NOT:
  - Verify domain equation values or evaluation metric algebra
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import torch
from support import configs
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src import domain, learning
from src.datasets.runtime import dataset_runtime_units as units

if TYPE_CHECKING:
    from collections.abc import Sequence

_MIDPOINT_EPOCH = 2
_TASK = domain.tasks.registry.get_task("steady_flow")
_GRID_SPACING = (0.003, 0.004)


class _ListMappingDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Present caller-supplied tensor mappings to a DataLoader in their original order.

    The fake owns no storage, normalization, or task semantics. Tests choose batch
    sizes over its finite samples to expose partial-batch aggregation behavior.
    """

    def __init__(self, samples: list[dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


@dataclass(frozen=True)
class _StateGroup:
    """Expose state channel names and Train scales like a scaling group."""

    names: tuple[str, ...]
    scale: torch.Tensor


@dataclass(frozen=True)
class _StateScales:
    """Expose Train state scales for synthetic physical-space tests; inputs and states decode unchanged."""

    names: tuple[str, ...]
    values: tuple[float, ...]

    def decode_inputs(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return value

    def decode_state(self, value: torch.Tensor) -> torch.Tensor:
        """Return the state tensor unchanged."""
        return value

    def group(self, name: str) -> _StateGroup:
        """Return the state group."""
        assert name == "state"
        return _StateGroup(self.names, torch.tensor(self.values, dtype=torch.float64))


class IdentityScaling:
    """Decode tensors unchanged for synthetic physical-space tests; every state has Train scale one."""

    def decode_inputs(self, value: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return value

    def decode_state(self, value: torch.Tensor) -> torch.Tensor:
        """Return the state tensor unchanged."""
        return value

    def group(self, name: str) -> _StateGroup:
        """Return unit Train scales for the steady states."""
        return _StateScales(_TASK.output_names, (1.0,) * len(_TASK.output_names)).group(name)


class ChannelAffineNormalizer:
    """
    Apply per-channel affine statistics to float64 BCHW tensors.

    The statistics are test-owned and deliberately differ from the identity, so
    a gradient reaching the normalized prediction must cross a nontrivial scale.
    """

    def __init__(self, statistics: Sequence[tuple[float, float]]) -> None:
        """Store one ``(mean, standard_deviation)`` pair per channel."""
        self.mean = torch.tensor([mean for mean, _ in statistics], dtype=torch.float64).view(1, -1, 1, 1)
        self.standard_deviation = torch.tensor([deviation for _, deviation in statistics], dtype=torch.float64).view(1, -1, 1, 1)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize one tensor."""
        return (tensor - self.mean) / self.standard_deviation

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the physical tensor of a normalized one."""
        return tensor * self.standard_deviation + self.mean


class AffineScaling:
    """Decode inputs and states through test-owned per-channel affines."""

    def __init__(self, inputs: ChannelAffineNormalizer, outputs: ChannelAffineNormalizer) -> None:
        """Store the input and state affines."""
        self.inputs = inputs
        self.outputs = outputs

    def decode_inputs(self, value: torch.Tensor) -> torch.Tensor:
        """Return physical inputs."""
        return self.inputs.inverse_transform(value)

    def decode_state(self, value: torch.Tensor) -> torch.Tensor:
        """Return physical states."""
        return self.outputs.inverse_transform(value)

    def group(self, name: str) -> _StateGroup:
        """Return the state affine's scales as the Train state scales."""
        return _StateScales(_TASK.output_names, tuple(self.outputs.standard_deviation.flatten().tolist())).group(name)


_TARGET_WEIGHTS = {"momentum": 2.0, "continuity": 4.0, "boundary": 3.0}


def _physics_config(*, model_kind: str = "fno") -> dict[str, object]:
    """
    Resolve a PI recipe with distinct term targets, four-epoch linear warmups, and crop one.

    Distinct momentum, continuity, and boundary targets make every midpoint weight
    observable while retaining the selected FNO or UNO recipe.
    """
    raw = configs.direct_config(
        model_kind=model_kind,
        physics_enabled=True,
    )
    for term, target in _TARGET_WEIGHTS.items():
        raw["loss"]["physics"][f"{term}_weight"] = {  # type: ignore[index]
            "target": target,
            "warmup": {"kind": "linear", "epochs": 4},
        }
    raw["loss"]["physics"]["interior_crop"] = 1  # type: ignore[index]
    return configs.resolve(raw)


def _manufactured_batch(*, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one TaskSpec-ordered manufactured steady-flow batch."""
    # Both axes hold at least the 10 points spectral Fourier continuation needs.
    height = 13
    width = 11
    y_values = torch.linspace(0.0, 1.0, height, dtype=dtype)
    x_values = torch.linspace(0.0, 2.0, width, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_values, x_values, indexing="ij")
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)
    zeros = torch.zeros_like(x_grid)
    p_in_bc = zeros.clone()
    p_in_bc[:, 0, :] = 1.0

    coordinate_values = iter((x_grid, y_grid))
    input_by_field: dict[str, torch.Tensor] = {}
    for field in _TASK.inputs:
        if field.role == "coordinate":
            input_by_field[field.name] = next(coordinate_values)
        elif field.name == "Kxy":
            input_by_field[field.name] = zeros
        elif field.role == "permeability":
            # log10 of 1e-8 m², a bed-like permeability with an order-one Darcy scale.
            input_by_field[field.name] = torch.full_like(x_grid, -8.0)
        elif field.role == "porosity":
            input_by_field[field.name] = torch.full_like(x_grid, 0.5)
        elif field.role == "boundary":
            input_by_field[field.name] = p_in_bc
        else:
            msg = f"unsupported steady-flow fixture role: {field.role}"
            raise AssertionError(msg)
    inputs = torch.stack([input_by_field[field] for field in _TASK.input_names], dim=1)

    velocity = next(group for group in _TASK.output_groups if group.id == "velocity")
    prediction_by_field = dict.fromkeys(_TASK.output_names, zeros)
    prediction_by_field[velocity.fields[0]] = 1e-5 * x_grid
    # A height-dependent v gives every cross-section a different volume flux.
    prediction_by_field[velocity.fields[1]] = 1e-5 * y_grid
    prediction = torch.stack([prediction_by_field[field] for field in _TASK.output_names], dim=1)
    target = torch.zeros_like(prediction)
    return inputs, prediction, target


def _batch_components(loss: Any, pred: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
    """Return every named component averaged over the equally sized cases of one batch."""
    return {name: value.mean() for name, value in loss.unit_components(pred, **kwargs).items()}


def test_disabled_physics_exposes_named_zero_components() -> None:
    """
    Compute a supervised loss from nonzero predictions with physics disabled.

    Named physics components must remain present as exact zeros and total must
    equal data loss, keeping telemetry schema stable without fabricating residuals.
    """
    supervised = configs.resolve(
        configs.direct_config(model_kind="fno", physics_enabled=False),
    )
    supervised_loss = learning.losses.factory.build_training_loss(supervised, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
    supervised_loss.set_scaling(IdentityScaling())
    task = domain.tasks.registry.get_task(str(supervised["task"]))
    pred = torch.ones((2, task.out_channels, 8, 8))
    target = torch.zeros_like(pred)
    components = _batch_components(supervised_loss, pred, x=None, y=target)

    assert tuple(components) == ("total", "data", "data_p", "data_velocity", "data_u", "data_v", "momentum", "continuity", "boundary")
    assert tuple(components) == supervised_loss.component_names
    assert components["total"] == components["data"]
    for term in ("momentum", "continuity", "boundary"):
        assert components[term].item() == 0.0
    assert supervised_loss.telemetry_state() == {}


def test_physics_composition_weighting_and_linear_warmup() -> None:
    """
    Evaluate one manufactured PI batch at warmup epochs zero, two, and four.

    Applied weights and named components must scale linearly and sum exactly,
    while explicit epoch state persists for deterministic resume.
    """
    loss = learning.losses.factory.build_training_loss(_physics_config(), device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
    assert isinstance(loss, learning.losses.pino.SemanticComposedLoss)
    loss.set_scaling(IdentityScaling())
    inputs, pred, target = _manufactured_batch()

    start = _batch_components(loss, pred, x=inputs, y=target, epoch=0)
    midpoint = _batch_components(loss, pred, x=inputs, y=target, epoch=2)
    end = _batch_components(loss, pred, x=inputs, y=target, epoch=4)
    per_case = loss.unit_components(pred, x=inputs, y=target, epoch=4)
    assert all(value.shape == (pred.shape[0],) for value in per_case.values())
    # A physics term is the batch mean given to every case, so its case mean is the batch value.
    assert torch.equal(per_case["momentum"], per_case["momentum"][:1].expand(pred.shape[0]))

    for term in ("momentum", "continuity", "boundary"):
        assert start[term].item() == 0.0
        assert end[term].item() > 0.0
        assert midpoint[term] == pytest.approx(0.5 * end[term])
    for components in (start, midpoint, end):
        assert components["total"] == pytest.approx(components["data"] + components["momentum"] + components["continuity"] + components["boundary"])
    assert loss.component_weights(epoch=2) == {"data": 1.0, **{term: 0.5 * target for term, target in _TARGET_WEIGHTS.items()}}
    assert loss.telemetry_state(epoch=2) == {
        key: value for term, target in _TARGET_WEIGHTS.items() for key, value in ((f"weight_{term}", 0.5 * target), (f"warmup_{term}_fraction", 0.5))
    }

    loss.set_epoch(_MIDPOINT_EPOCH)
    forward_total = loss(pred, x=inputs, y=target)
    assert forward_total == pytest.approx(midpoint["total"])
    assert int(loss.state_dict()["current_epoch"].item()) == _MIDPOINT_EPOCH
    with pytest.raises(ValueError, match="non-negative integer"):
        loss.set_epoch(-1)


def _constant_errors(errors: Sequence[float], *, shape: tuple[int, int] = (5, 7)) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a two-sample prediction and target whose Train-scaled error is one constant per channel."""
    target = torch.randn((2, len(errors), *shape), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    offsets = torch.tensor(errors, dtype=torch.float64).view(1, -1, 1, 1)
    return target + offsets, target


def test_steady_state_loss_weights_pressure_and_the_velocity_vector_by_half() -> None:
    """
    Score constant scaled errors with distinct Train scales.

    Pressure and the velocity vector each weigh 1/2; the vector term divides the summed
    physical squared errors by the summed squared Train scales, (E_u + E_v) / (s_u^2 + s_v^2).
    """
    scales = {"p": 2.0, "u": 3.0, "v": 4.0}
    errors = {"p": 0.5, "u": 1.5, "v": -2.0}
    loss = learning.losses.factory.build_data_loss(_TASK, "mse")
    loss.set_scaling(_StateScales(_TASK.output_names, tuple(scales[name] for name in _TASK.output_names)))
    prediction, target = _constant_errors([errors[name] for name in _TASK.output_names])

    physical = {name: (scales[name] * errors[name]) ** 2 for name in scales}
    velocity = (physical["u"] + physical["v"]) / (scales["u"] ** 2 + scales["v"] ** 2)
    contributions = loss.term_contributions(prediction, target)

    assert set(contributions) == {"p", "velocity"}
    assert float(contributions["p"]) == pytest.approx(0.5 * errors["p"] ** 2)
    assert float(contributions["velocity"]) == pytest.approx(0.5 * velocity)
    assert float(loss(prediction, target)) == pytest.approx(0.5 * errors["p"] ** 2 + 0.5 * velocity)
    unscaled = learning.losses.factory.build_data_loss(_TASK, "mse")
    with pytest.raises(RuntimeError, match="set_scaling"):
        unscaled(prediction, target)


def test_steady_data_parts_split_the_data_loss_without_changing_the_total() -> None:
    """
    Split the weighted data loss into p and velocity and the velocity into its u and v shares.

    Each field's share is its physical squared error over the summed squared Train scales of the
    vector, times the term weight 1/2; the parts sum to their parent up to float rounding.
    """
    scales = {"p": 2.0, "u": 3.0, "v": 4.0}
    errors = {"p": 0.5, "u": 1.5, "v": -2.0}
    data_weight = 0.7
    state_scales = _StateScales(_TASK.output_names, tuple(scales[name] for name in _TASK.output_names))
    state_loss = learning.losses.factory.build_data_loss(_TASK, "mse")
    state_loss.set_scaling(state_scales)
    prediction, target = _constant_errors([errors[name] for name in _TASK.output_names])

    contributions, shares = state_loss.unit_contributions_with_shares(prediction, target)

    vector_scale = scales["u"] ** 2 + scales["v"] ** 2
    assert set(shares) == {"velocity"}
    for field in ("u", "v"):
        expected = 0.5 * (scales[field] * errors[field]) ** 2 / vector_scale
        assert torch.allclose(shares["velocity"][field], torch.full_like(shares["velocity"][field], expected))
    assert torch.allclose(shares["velocity"]["u"] + shares["velocity"]["v"], contributions["velocity"])
    assert all(torch.equal(value, contributions[name]) for name, value in state_loss.unit_term_contributions(prediction, target).items())

    request = configs.direct_config(model_kind="fno", physics_enabled=False)
    request["loss"]["data"]["weight"] = data_weight
    loss = learning.losses.factory.build_training_loss(configs.resolve(request), device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
    loss.set_scaling(state_scales)
    components = loss.unit_components(prediction, x=None, y=target)
    own_terms, own_shares = loss.data_loss.unit_contributions_with_shares(prediction, target)

    assert tuple(components) == loss.component_names
    for name, value in own_terms.items():
        assert torch.allclose(components[f"data_{name}"], data_weight * value)
    for field, value in own_shares["velocity"].items():
        assert torch.allclose(components[f"data_{field}"], data_weight * value)
    assert not torch.allclose(components["data_u"], components["data_v"])
    assert torch.allclose(components["data_p"] + components["data_velocity"], components["data"])
    assert torch.allclose(components["data_u"] + components["data_v"], components["data_velocity"])
    assert torch.equal(components["total"], components["data"])


def test_transient_state_loss_weights_temperature_humidity_and_both_moisture_compartments() -> None:
    """
    Score constant scaled state errors of the drying states.

    Temperature and humidity weigh 1/3 each and the two moisture compartments 1/6 each,
    with one named data component per term.
    """
    task = domain.tasks.registry.get_task("transient_drying")
    errors = {"T": 1.0, "phi": 2.0, "w_surf": 3.0, "w_int": 4.0}
    weights = {"T": 1 / 3, "phi": 1 / 3, "w_surf": 1 / 6, "w_int": 1 / 6}
    loss = learning.losses.transient.TransientStateLoss(data_loss=learning.losses.factory.build_data_loss(task, "mse"), data_weight=2.0)
    prediction, target = _constant_errors([errors[name] for name in task.metric_names])

    components = {name: value.mean() for name, value in loss.unit_components(prediction, y=target).items()}

    assert loss.component_names == ("total", "data", "data_T", "data_phi", "data_w_surf", "data_w_int")
    for name, error in errors.items():
        assert float(components[f"data_{name}"]) == pytest.approx(2.0 * weights[name] * error**2)
    assert float(components["total"]) == pytest.approx(2.0 * sum(weights[name] * error**2 for name, error in errors.items()))


def test_h1_adds_the_weighted_squared_gradient_error_in_metres_without_wrap() -> None:
    """
    Add linear errors whose physical gradients are exact on a non-square anisotropic grid.

    The H1 loss exceeds the MSE loss by the weighted squared gradient in metres (length scale 1 m);
    a periodic derivative would add the end-to-end jump, and swapped spacings change the value.
    """
    task = domain.tasks.registry.get_task("transient_drying")
    spacing = (0.5, 0.25)
    rows, columns = torch.meshgrid(torch.arange(6, dtype=torch.float64), torch.arange(9, dtype=torch.float64), indexing="ij")
    y, x = rows * spacing[0], columns * spacing[1]
    slopes = {"T": (1.0, 0.0), "phi": (0.0, 2.0), "w_surf": (3.0, -1.0), "w_int": (0.5, 0.5)}
    error = torch.stack([slope_y * y + slope_x * x for slope_y, slope_x in slopes.values()])[None]
    target = torch.zeros_like(error)
    weights = {"T": 1 / 3, "phi": 1 / 3, "w_surf": 1 / 6, "w_int": 1 / 6}
    gradient = sum(weights[name] * (slope_y**2 + slope_x**2) for name, (slope_y, slope_x) in slopes.items())

    mse = learning.losses.factory.build_data_loss(task, "mse")
    h1 = learning.losses.factory.build_data_loss(task, "h1", grid_spacing=spacing)
    swapped = learning.losses.factory.build_data_loss(task, "h1", grid_spacing=spacing[::-1])

    assert float(h1(error, target)) == pytest.approx(float(mse(error, target)) + gradient, rel=1e-12)
    assert float(swapped(error, target)) != pytest.approx(float(h1(error, target)), rel=1e-3)
    with pytest.raises(ValueError, match="two positive finite grid spacings"):
        learning.losses.factory.build_data_loss(task, "h1")


def test_pi_model_families_share_one_dimensionless_loss_path() -> None:
    """
    Build the PI loss for both model families and compare it with the domain terms.

    Model kind may vary, but every weighted component must be its target weight
    times the matching dimensionless domain diagnostic, with no second formula.
    """
    for model_kind in ("fno", "uno"):
        config = _physics_config(model_kind=model_kind)
        loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
        loss.set_scaling(IdentityScaling())
        inputs, pred, target = _manufactured_batch()

        diagnostics = loss.compute_physics_diagnostics(pred, x=inputs)
        components = _batch_components(loss, pred, x=inputs, y=target, epoch=4)
        expected = {
            "momentum": diagnostics.momentum_mse,
            "continuity": diagnostics.continuity_mse,
            "boundary": diagnostics.boundary_mse,
        }

        assert config["model"]["kind"] == model_kind  # type: ignore[index]
        assert type(loss) is learning.losses.pino.SemanticComposedLoss
        assert loss.component_names == tuple(components)
        for term, value in expected.items():
            assert torch.allclose(components[term], _TARGET_WEIGHTS[term] * value)


@pytest.mark.parametrize("term", tuple(_TARGET_WEIGHTS))
def test_each_physics_term_backpropagates_through_the_normalizers_to_the_prediction(term: str) -> None:
    """
    Differentiate one weighted physics term at a float64 prediction with affine normalizers.

    The autograd derivative along a fixed direction must equal a central finite
    difference of the same term, so no term is detached from the prediction or
    loses the normalization scale on its way back to the model output.
    """
    loss = learning.losses.factory.build_training_loss(_physics_config(), device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
    # Distinct non-identity means and scales per channel; velocities are given m/s-sized scales.
    input_statistics = [(0.3 + 0.2 * index, 1.5 + 0.5 * index) for index, _ in enumerate(_TASK.input_names)]
    output_statistics = {"p": (0.1, 0.4), "u": (1e-6, 2e-5), "v": (-2e-6, 3e-5)}
    in_normalizer = ChannelAffineNormalizer(input_statistics)
    out_normalizer = ChannelAffineNormalizer([output_statistics[name] for name in _TASK.output_names])
    loss.set_scaling(AffineScaling(in_normalizer, out_normalizer))
    physical_inputs, physical_prediction, target = _manufactured_batch(dtype=torch.float64)
    inputs = in_normalizer.transform(physical_inputs)
    prediction = out_normalizer.transform(physical_prediction).requires_grad_(True)
    direction = torch.randn(prediction.shape, generator=torch.Generator().manual_seed(11), dtype=torch.float64)

    def weighted_term(value: torch.Tensor) -> torch.Tensor:
        return _batch_components(loss, value, x=inputs, y=target, epoch=4)[term]

    component = weighted_term(prediction)
    assert component.requires_grad
    (gradient,) = torch.autograd.grad(component, prediction)
    step = 1e-3
    with torch.no_grad():
        numeric = (weighted_term(prediction + step * direction) - weighted_term(prediction - step * direction)) / (2.0 * step)

    # The reference must be informative: a zero slope would let a detached term pass.
    assert abs(numeric.item()) > 1e-6
    assert (gradient * direction).sum().item() == pytest.approx(numeric.item(), rel=1e-6)


def test_invalid_physics_loss_settings_fail_through_public_factory() -> None:
    """
    Replace one valid derivative kind and one warmup kind with unsupported values.

    The public loss factory must reject each semantic setting before training,
    preventing implicit numerical or schedule fallbacks.
    """
    derivative_config = _physics_config()
    derivative_config["loss"]["physics"]["derivatives"]["kind"] = "unsupported"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown derivative identifier"):
        learning.losses.factory.build_training_loss(derivative_config, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)

    warmup_config = _physics_config()
    warmup_config["loss"]["physics"]["continuity_weight"]["warmup"]["kind"] = "cosine"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown warmup identifier"):
        learning.losses.factory.build_training_loss(warmup_config, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)


def _steady_update(memory_batch: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Train one steady PI update over four cases split into microbatches of ``memory_batch``; return the stepped weights."""
    inputs, _pred, _target = _manufactured_batch(dtype=torch.float64)
    targets = 1e-5 * torch.randn((4, len(_TASK.output_names), *inputs.shape[-2:]), generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    samples = [{"x": inputs[0], "y": target} for target in targets]
    sampler = units.UnitSampler(unit_kind="case", cases=range(4), keys=None, units_per_update=4, memory_batch=memory_batch, seed=0)
    sampler.begin_epoch(0)
    loader = DataLoader(_ListMappingDataset(samples), batch_sampler=sampler)
    model = torch.nn.Conv2d(len(_TASK.input_names), len(_TASK.output_names), 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(1e-6 * torch.randn(model.weight.shape, generator=torch.Generator().manual_seed(7), dtype=torch.float64))
    loss = learning.losses.factory.build_training_loss(_physics_config(), device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
    loss.set_scaling(IdentityScaling())
    # Four committed passes over the four cases end the four-epoch warm-up, so the update applies the target weights.
    values = learning.training.loop.train_update_groups(
        model,
        loader,
        SGD(model.parameters(), lr=1.0),
        loss,
        torch.device("cpu"),
        learning.training.loop._SteadyUnits(None, torch.device("cpu")),
        budget=learning.training.budget.StageBudget(unit="units", total=20, units=16, validated_point=16),
    )
    return model.weight.detach().clone(), values


def test_steady_update_gradient_does_not_depend_on_the_memory_split() -> None:
    """
    Train one update of four cases at once and case by case with the PI loss.

    The case losses and the batch-mean physics terms enter divided by the planned
    four units, so a 1 x 4 and a 4 x 1 memory split give the same step, and the
    epoch telemetry reports the units, cases and physics terms of the update.
    """
    whole, values = _steady_update(4)
    split, split_values = _steady_update(1)

    assert torch.allclose(whole, split, rtol=1e-10, atol=1e-18)
    assert split_values["train/loss_total"] == pytest.approx(values["train/loss_total"], rel=1e-10)
    assert (values["train/updates"], values["train/microbatches"], split_values["train/microbatches"]) == (1.0, 1.0, 4.0)
    assert (values["train/units_per_update"], values["train/distinct_cases_per_update"], values["gradient_predictions"]) == (4.0, 4.0, 4.0)
    assert values["physics/train/loss_momentum"] > 0.0
    assert values["physics/train/continuity_weight"] == _TARGET_WEIGHTS["continuity"]


class _ManufacturedMonitorModel(torch.nn.Module):
    """Return a deterministic physical-as-normalized velocity field."""

    def __init__(self) -> None:
        super().__init__()
        self.samples_seen = 0
        self.inference_modes: list[bool] = []
        self.grad_modes: list[bool] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return manufactured outputs and record bounded inference isolation."""
        self.samples_seen += int(inputs.shape[0])
        self.inference_modes.append(torch.is_inference_mode_enabled())
        self.grad_modes.append(torch.is_grad_enabled())
        x_values = inputs[:, _TASK.input_names.index("x")]
        y_values = inputs[:, _TASK.input_names.index("y")]
        zeros = torch.zeros_like(x_values)
        velocity = next(group for group in _TASK.output_groups if group.id == "velocity")
        prediction_by_field = dict.fromkeys(_TASK.output_names, zeros)
        prediction_by_field[velocity.fields[0]] = 1e-5 * x_values
        prediction_by_field[velocity.fields[1]] = 1e-5 * y_values
        return torch.stack([prediction_by_field[field] for field in _TASK.output_names], dim=1)


def test_physics_monitor_is_bounded_and_reports_the_three_terms() -> None:
    """
    Evaluate the shared domain diagnostics on only a fixed evaluation prefix.

    The monitor must consume exactly the bound and report the dimensionless
    momentum, flux-balance, and boundary values without a second physics formula.
    """
    inputs, _pred, target = _manufactured_batch()
    samples = [{"x": inputs[0].clone(), "y": target[0].clone()} for _ in range(3)]
    loader = DataLoader(_ListMappingDataset(samples), batch_size=3, shuffle=False)

    for model_kind in ("fno", "uno"):
        for physics_enabled in (False, True):
            config = configs.resolve(
                configs.direct_config(
                    model_kind=model_kind,
                    physics_enabled=physics_enabled,
                ),
            )
            loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
            loss.set_scaling(IdentityScaling())
            model = _ManufacturedMonitorModel()

            values = learning.training.loop.evaluate_physics_monitor(
                model,
                loader,
                loss,
                torch.device("cpu"),
                scaling=None,
                max_cases=2,
            )

            assert model.samples_seen == 2
            assert model.inference_modes == [True]
            assert model.grad_modes == [False]
            assert set(values) == {
                "physics/id/momentum_residual_mse",
                "physics/id/flux_balance_mse",
                "physics/id/pressure_boundary_mse",
            }
            assert all(torch.isfinite(torch.tensor(value)) for value in values.values())
            assert values["physics/id/flux_balance_mse"] > 0.0


class _FailingMonitorLoss(torch.nn.Module):
    """Inject the demonstrated spacing owner below the monitor boundary."""

    def compute_physics_diagnostics(self, _pred: torch.Tensor, *, x: torch.Tensor) -> None:
        del x
        msg = "x-coordinate spacing is not uniform within tolerance"
        raise ValueError(msg)


def test_physics_monitor_wraps_numerics_as_scientific_evaluation_with_cause() -> None:
    """Keep numerical monitor failures scientific and restore model state."""
    inputs, _pred, target = _manufactured_batch()
    loader = DataLoader(
        _ListMappingDataset([{"x": inputs[0], "y": target[0]}]),
        batch_size=1,
        shuffle=False,
    )
    model = _ManufacturedMonitorModel()
    model.train()
    with pytest.raises(learning.training.loop.PhysicsMonitorEvaluationError) as captured:
        learning.training.loop.evaluate_physics_monitor(
            model,
            loader,
            _FailingMonitorLoss(),
            torch.device("cpu"),
            scaling=None,
            max_cases=1,
        )
    assert isinstance(captured.value.__cause__, ValueError)
    assert "x-coordinate spacing" in str(captured.value.__cause__)
    assert model.training
