"""Protect semantic supervised/physics loss composition and deterministic warmup.

Responsibilities:
  - Check named supervised/physics components, warmup, normalization, telemetry, and monitor
    failures
  - Check that the relative H1 data loss treats boundaries as non-periodic and differentiates in metres
  - Check that every physics term carries gradients through the normalizers to the prediction

Design principles:
  - Actual sample counts weight epoch evidence; disabled physics has explicit zero components

This module does NOT:
  - Verify domain equation values or evaluation metric algebra
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from support import configs
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader, Dataset

from src import domain, learning

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


class IdentityNormalizer:
    """Return tensors unchanged for synthetic physical-space tests."""

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the input tensor unchanged."""
        return tensor


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
    task = domain.tasks.registry.get_task(str(supervised["task"]))
    pred = torch.ones((2, task.out_channels, 8, 8))
    target = torch.zeros_like(pred)
    components = supervised_loss.compute_components(pred, x=None, y=target)

    assert tuple(components) == ("total", "data", "momentum", "continuity", "boundary")
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
    normalizer = IdentityNormalizer()
    loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
    inputs, pred, target = _manufactured_batch()

    start = loss.compute_components(pred, x=inputs, y=target, epoch=0)
    midpoint = loss.compute_components(pred, x=inputs, y=target, epoch=2)
    end = loss.compute_components(pred, x=inputs, y=target, epoch=4)

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
    assert forward_total == loss.last_components["total"]
    assert forward_total == pytest.approx(midpoint["total"])
    assert int(loss.state_dict()["current_epoch"].item()) == _MIDPOINT_EPOCH
    with pytest.raises(ValueError, match="non-negative integer"):
        loss.set_epoch(-1)


def test_relative_h1_ignores_the_wrap_between_opposite_boundaries() -> None:
    """
    Score the same interior error on a ramp and on a tent with equal slope and mean square.

    A periodic H1 would add the ramp's end-to-end jump to its gradient norm and
    shrink its relative error several-fold; the non-periodic norm scores both alike.
    """
    size = 64
    y_grid = torch.linspace(0.0, 1.0, size, dtype=torch.float64)[:, None].expand(size, size)
    ramp = y_grid
    # Offset chosen so the tent's mean square matches the ramp's 1/3.
    triangle = (0.5 + math.sqrt(1.25)) / 2.0 - (y_grid - 0.5).abs()
    bump = 0.01 * torch.exp(-(((y_grid - 0.5) / 0.1) ** 2))
    loss = learning.losses.factory.build_data_loss(
        "relative_h1", space="normalized", operator_dimensionality=2, grid_spacing=(1.0 / size, 1.0 / size)
    )

    def relative_h1(target: torch.Tensor) -> float:
        return float(loss((target + bump)[None, None], target[None, None]))

    assert relative_h1(ramp) == pytest.approx(relative_h1(triangle), rel=0.05)


def _quadratic_relative_h1_reference(spacing_y: float, spacing_x: float, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Return a quadratic target, its perturbed prediction and the analytic relative H1 on that physical grid.

    Quadratics are differentiated exactly by the central and the third-order one-sided stencils, so
    the analytic gradients in metres equal the discrete ones at every grid point.
    """
    height, width = shape
    y = (torch.arange(height, dtype=torch.float64) * spacing_y)[:, None].expand(height, width)
    x = (torch.arange(width, dtype=torch.float64) * spacing_x)[None, :].expand(height, width)
    target = 1.0 + 2.0 * y + 3.0 * x + y**2 + x * y
    error = 0.1 * (y**2 + 2.0 * x * y + x)
    target_fields = (target, 2.0 + 2.0 * y + x, 3.0 + y)
    error_fields = (error, 0.1 * (2.0 * y + 2.0 * x), 0.1 * (2.0 * y + 1.0))
    numerator = sum(float((field**2).sum()) for field in error_fields)
    denominator = sum(float((field**2).sum()) for field in target_fields)
    return target, target + error, math.sqrt(numerator) / math.sqrt(denominator)


def test_relative_h1_differentiates_in_metres_rows_then_columns() -> None:
    """
    Compare the factory loss with the analytic relative H1 of a quadratic on a non-square anisotropic grid.

    The spacing list is ordered (rows, columns); swapping it must change the value, so the loss
    cannot silently differentiate on a unit square or with transposed axes.
    """
    spacing_y, spacing_x = 0.5, 0.25
    target, prediction, expected = _quadratic_relative_h1_reference(spacing_y, spacing_x, (6, 9))
    pair = (prediction[None, None], target[None, None])

    loss = learning.losses.factory.build_data_loss("relative_h1", space="normalized", operator_dimensionality=2, grid_spacing=(spacing_y, spacing_x))
    swapped = learning.losses.factory.build_data_loss(
        "relative_h1", space="normalized", operator_dimensionality=2, grid_spacing=(spacing_x, spacing_y)
    )

    assert float(loss(*pair)) == pytest.approx(expected, rel=1e-9)
    assert float(swapped(*pair)) != pytest.approx(expected, rel=1e-3)


def test_relative_h1_requires_a_valid_physical_spacing() -> None:
    """Reject a missing spacing for relative H1 and malformed spacings, while relative L2 ignores it."""
    factory = learning.losses.factory
    with pytest.raises(ValueError, match="requires grid_spacing"):
        factory.build_data_loss("relative_h1", space="normalized", operator_dimensionality=2)
    for invalid in ((0.003,), (0.003, float("nan"))):
        with pytest.raises(ValueError, match="positive finite spacings"):
            factory.build_relative_h1(2, reduction="mean", grid_spacing=invalid)
    assert factory.build_data_loss("relative_l2", space="normalized", operator_dimensionality=2) is not None


def test_relative_h1_metric_uses_the_loss_spacing() -> None:
    """
    Evaluate one sample through the steady metric and through the factory with the same spacing.

    The metric averages the per-field values of its single sample like the data loss, so
    identical fields give the loss value; the metric also requires a spacing at construction.
    """
    spacing = (0.5, 0.25)
    target, prediction, expected = _quadratic_relative_h1_reference(*spacing, (6, 9))
    metric_id = "normalized_relative_h1"
    config = configs.resolve(configs.direct_config())
    config["evaluation"]["metrics"] = [metric for metric in config["evaluation"]["metrics"] if metric["id"] == metric_id]
    cpu = torch.device("cpu")
    channels = len(_TASK.output_names)
    pred = prediction.expand(1, channels, *prediction.shape)
    true = target.expand(1, channels, *target.shape)

    with pytest.raises(ValueError, match="requires grid_spacing"):
        learning.metrics.metrics.build_evaluation_metrics(config, device=cpu)
    metric = learning.metrics.metrics.build_evaluation_metrics(config, device=cpu, grid_spacing=spacing)[metric_id]
    metric.update(pred, true, space="normalized", batch_index=0)

    assert metric.compute() == pytest.approx(expected, rel=1e-9)


def test_pi_model_families_share_one_dimensionless_loss_path() -> None:
    """
    Build the PI loss for both model families and compare it with the domain terms.

    Model kind may vary, but every weighted component must be its target weight
    times the matching dimensionless domain diagnostic, with no second formula.
    """
    for model_kind in ("fno", "uno"):
        config = _physics_config(model_kind=model_kind)
        loss = learning.losses.factory.build_training_loss(config, device=torch.device("cpu"), grid_spacing=_GRID_SPACING)
        normalizer = IdentityNormalizer()
        loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
        inputs, pred, target = _manufactured_batch()

        diagnostics = loss.compute_physics_diagnostics(pred, x=inputs)
        components = loss.compute_components(pred, x=inputs, y=target, epoch=4)
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
    loss.set_normalizers(in_normalizer=in_normalizer, out_normalizer=out_normalizer)
    physical_inputs, physical_prediction, target = _manufactured_batch(dtype=torch.float64)
    inputs = in_normalizer.transform(physical_inputs)
    prediction = out_normalizer.transform(physical_prediction).requires_grad_(True)
    direction = torch.randn(prediction.shape, generator=torch.Generator().manual_seed(11), dtype=torch.float64)

    def weighted_term(value: torch.Tensor) -> torch.Tensor:
        return loss.compute_components(value, x=inputs, y=target, epoch=4)[term]

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


class _IdentityScaleModel(torch.nn.Module):
    """
    Expose one trainable scalar for epoch-aggregation tests.

    Callers use zero learning rate so predictions reveal only batch weighting.
    this helper is not a model-accuracy fixture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Scale a synthetic batch while retaining an autograd path."""
        return self.scale * value


class _SyntheticComponentLoss(torch.nn.Module):
    """Return batch-mean components whose unequal final batch is observable."""

    physics_enabled = False

    def __init__(self) -> None:
        super().__init__()
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch mean and expose detached total/data components."""
        del x
        if y is None:
            msg = "target required"
            raise ValueError(msg)
        value = (pred - y).mean()
        self.last_components = {"total": value.detach(), "data": value.detach()}
        return value


class _SyntheticPhysicsComponentLoss(_SyntheticComponentLoss):
    """Expose the three physics components and fixed applied-weight telemetry."""

    physics_enabled = True

    def forward(
        self,
        pred: torch.Tensor,
        y: torch.Tensor | None = None,
        *,
        x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fixed-ratio named physics components from one batch mean."""
        del x
        if y is None:
            msg = "target required"
            raise ValueError(msg)
        data = (pred - y).mean()
        momentum = 2.0 * data
        boundary = 3.0 * data
        continuity = 4.0 * data
        total = data + momentum + boundary + continuity
        self.last_components = {
            "total": total.detach(),
            "data": data.detach(),
            "momentum": momentum.detach(),
            "boundary": boundary.detach(),
            "continuity": continuity.detach(),
        }
        return total

    def telemetry_state(self) -> dict[str, float]:
        """Return fixed applied-weight telemetry for epoch aggregation."""
        return {
            "weight_momentum": 0.25,
            "warmup_momentum_fraction": 0.75,
            "weight_continuity": 0.125,
            "warmup_continuity_fraction": 0.75,
            "weight_boundary": 0.5,
            "warmup_boundary_fraction": 1.0,
        }


def test_epoch_loss_components_are_weighted_by_actual_sample_count() -> None:
    """
    Train over a two-sample batch followed by one unequal-valued singleton.

    Every named component and fixed physics telemetry must use actual sample count,
    preventing a partial batch from receiving equal weight to a full batch.
    """
    samples = [{"x": torch.full((1, 1, 1), value), "y": torch.zeros((1, 1, 1))} for value in (1.0, 3.0, 9.0)]
    loader = DataLoader(_ListMappingDataset(samples), batch_size=2, shuffle=False)
    model = _IdentityScaleModel()
    optimizer = SGD(model.parameters(), lr=0.0)

    values = learning.training.loop.train_one_epoch(
        model,
        loader,
        optimizer,
        _SyntheticComponentLoss(),
        torch.device("cpu"),
    )

    assert values["train/loss_total"] == pytest.approx(13.0 / 3.0)
    assert values["train/loss_data"] == pytest.approx(13.0 / 3.0)
    assert values["train/loss_total"] != pytest.approx((2.0 + 9.0) / 2.0)
    assert values["optimizer_steps"] == 2.0
    assert values["system/train_duration_seconds"] > 0.0
    assert values["system/train_samples_per_second"] > 0.0
    assert not any("momentum" in key or "continuity" in key for key in values)

    physics_model = _IdentityScaleModel()
    physics_values = learning.training.loop.train_one_epoch(
        physics_model,
        loader,
        SGD(physics_model.parameters(), lr=0.0),
        _SyntheticPhysicsComponentLoss(),
        torch.device("cpu"),
    )
    assert set(physics_values) == {
        "train/loss_total",
        "train/loss_data",
        "physics/train/loss_momentum",
        "physics/train/loss_continuity",
        "physics/train/loss_boundary",
        "physics/train/momentum_weight",
        "physics/train/continuity_weight",
        "physics/train/boundary_weight",
        "system/train_duration_seconds",
        "system/train_samples_per_second",
        "optimizer_steps",
        "committed_optimizer_groups",
        "gradient_predictions",
    }
    assert physics_values["physics/train/loss_continuity"] == pytest.approx(4.0 * 13.0 / 3.0)
    assert physics_values["physics/train/continuity_weight"] == 0.125


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
            normalizer = IdentityNormalizer()
            loss.set_normalizers(in_normalizer=normalizer, out_normalizer=normalizer)
            model = _ManufacturedMonitorModel()

            values = learning.training.loop.evaluate_physics_monitor(
                model,
                loader,
                loss,
                torch.device("cpu"),
                data_processor=None,
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
            data_processor=None,
            max_cases=1,
        )
    assert isinstance(captured.value.__cause__, ValueError)
    assert "x-coordinate spacing" in str(captured.value.__cause__)
    assert model.training
