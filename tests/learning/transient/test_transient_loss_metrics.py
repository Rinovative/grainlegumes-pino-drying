# ruff: noqa: S101
"""Protect transient reconstructed-state metrics and scaled-increment loss."""

from __future__ import annotations

import math

import pytest
import torch

from src import domain, learning

_TASK = domain.tasks.registry.get_task("transient_drying")
_DRYING_METRIC = next(metric for metric in _TASK.default_metrics if metric.kind == "drying_group_macro_rmse")
_PHYSICAL_MAE = next(metric for metric in _TASK.default_metrics if metric.id == "physical_mae_T")
_BATCH_INDEX = 7
_HUBER_BETA = 0.5


def _metric(
    definition: domain.tasks.spec.MetricSpec,
) -> learning.metrics.metrics.DatasetMetric:
    """Build one test-selected transient metric on CPU."""
    config = {
        "task": _TASK.id,
        "evaluation": {
            "metrics": [
                definition.as_dict(all_fields=_TASK.metric_names),
            ],
        },
    }
    return learning.metrics.metrics.build_evaluation_metrics(
        config,
        device=torch.device("cpu"),
    )[definition.id]


def _loss_config(
    *,
    kind: str = "mse",
    beta: float = 1.0,
    channel_weights: list[float] | None = None,
    state_aux_weight: float = 0.0,
    physics_enabled: bool = False,
) -> dict[str, object]:
    """Return one minimal resolved transient loss selection."""
    return {
        "task": _TASK.id,
        "loss": {
            "data": {
                "kind": kind,
                "space": "scaled_increment",
                "weight": 1.0,
                "beta": beta,
                "channel_weights": ([1.0, 1.0, 1.0, 1.0] if channel_weights is None else channel_weights),
                "state_aux_weight": state_aux_weight,
            },
            "physics": {
                "enabled": physics_enabled,
                "continuity": "none",
            },
        },
    }


def test_drying_macro_uses_exact_weights_and_global_statistics() -> None:
    """Weight global field RMSE as 1/3, 1/3, 1/6, and 1/6."""
    metric = _metric(_DRYING_METRIC)
    target = torch.zeros(3, 4, 2, 2)
    field_errors = torch.tensor([3.0, 6.0, 9.0, 12.0])
    prediction = field_errors.view(1, 4, 1, 1).expand_as(target)
    metric.update(
        prediction,
        target,
        space="normalized",
        batch_index=_BATCH_INDEX,
    )
    expected = 3.0 / 3.0 + 6.0 / 3.0 + 9.0 / 6.0 + 12.0 / 6.0
    assert metric.compute() == pytest.approx(expected)

    partitioned = _metric(_DRYING_METRIC)
    for index, (pred_batch, target_batch) in enumerate(
        zip(
            prediction.split((1, 2)),
            target.split((1, 2)),
            strict=True,
        )
    ):
        partitioned.update(
            pred_batch,
            target_batch,
            space="normalized",
            batch_index=index,
        )
    assert partitioned.compute() == pytest.approx(expected)


def test_drying_macro_accumulates_masked_global_statistics() -> None:
    """Use valid-mask counts rather than finalized batch averages."""
    metric = _metric(_DRYING_METRIC)
    assert isinstance(metric, learning.metrics.metrics.DryingGroupMacroRMSEMetric)
    target = torch.zeros(2, 4, 1, 2)
    prediction = torch.zeros_like(target)
    prediction[0, :, 0, 0] = torch.tensor([3.0, 6.0, 9.0, 12.0])
    prediction[1, :, 0, 1] = torch.tensor([6.0, 12.0, 18.0, 24.0])
    mask = torch.tensor([[[[True, False]]], [[[False, True]]]])

    metric.update(prediction, target, space="normalized", batch_index=0, mask=mask)

    expected_fields = [
        math.sqrt((3.0**2 + 6.0**2) / 2.0),
        math.sqrt((6.0**2 + 12.0**2) / 2.0),
        math.sqrt((9.0**2 + 18.0**2) / 2.0),
        math.sqrt((12.0**2 + 24.0**2) / 2.0),
    ]
    expected = sum(weight * value for weight, value in zip((1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), expected_fields, strict=True))
    assert metric.compute() == pytest.approx(expected)
    assert metric.components()["grain_moisture_error"] == pytest.approx(
        0.5 * (expected_fields[2] + expected_fields[3]),
    )

    with pytest.raises(ValueError, match="no valid"):
        metric.update(
            prediction,
            target,
            space="normalized",
            batch_index=1,
            mask=torch.zeros_like(mask),
        )


def test_mae_accumulates_global_absolute_error() -> None:
    """Finalize transient MAE from total absolute error and element count."""
    metric = _metric(_PHYSICAL_MAE)
    first = torch.zeros(1, 4, 1, 2)
    second = torch.zeros_like(first)
    first[:, 0] = torch.tensor([1.0, -3.0])
    second[:, 0] = torch.tensor([5.0, -7.0])
    target = torch.zeros_like(first)
    metric.update(first, target, space="physical", batch_index=0)
    metric.update(second, target, space="physical", batch_index=1)
    assert metric.compute() == pytest.approx(4.0)


def test_mse_reduces_every_nonchannel_axis_with_explicit_weights() -> None:
    """Reduce BLCHW errors by channel before normalized weighted aggregation."""
    weights = [1.0, 1.0, 2.0, 2.0]
    loss = learning.losses.factory.build_training_loss(
        _loss_config(channel_weights=weights),
        device=torch.device("cpu"),
    )
    channel_errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
    prediction = channel_errors.view(1, 1, 4, 1, 1).expand(2, 3, 4, 2, 2)
    target = torch.zeros_like(prediction)
    value = loss(prediction, target)
    expected = sum(weight * error**2 for weight, error in zip(weights, channel_errors.tolist(), strict=True)) / sum(weights)
    assert value == pytest.approx(expected)
    assert tuple(loss.last_components) == ("total", "data", "data_T", "data_phi", "data_w_surf", "data_w_int", "state_aux")
    assert loss.last_components["state_aux"].item() == 0.0


def test_huber_and_optional_reconstructed_state_auxiliary_are_named() -> None:
    """Apply positive-beta Huber data loss plus disabled-by-default state loss."""
    loss = learning.losses.factory.build_training_loss(
        _loss_config(
            kind="huber",
            beta=_HUBER_BETA,
            state_aux_weight=0.25,
        ),
        device=torch.device("cpu"),
    )
    prediction = torch.ones(1, 2, 4, 1, 1)
    target = torch.zeros_like(prediction)
    predicted_state = torch.full_like(prediction, 2.0)
    target_state = torch.zeros_like(prediction)
    components = loss.compute_components(
        prediction,
        y=target,
        predicted_state=predicted_state,
        target_state=target_state,
    )
    expected_huber = _HUBER_BETA * (1.0 - 0.5 * _HUBER_BETA)
    assert components["data"] == pytest.approx(expected_huber)
    assert components["state_aux"] == pytest.approx(1.0)
    assert components["total"] == pytest.approx(expected_huber + 1.0)


def test_transient_loss_validation_skips_steady_physics_construction() -> None:
    """Reject unsupported physics and malformed loss values at task ownership."""
    loss = learning.losses.factory.build_training_loss(
        _loss_config(),
        device=torch.device("cpu"),
    )
    assert isinstance(
        loss,
        learning.losses.transient.TransientIncrementLoss,
    )
    assert loss.physics_enabled is False

    with pytest.raises(ValueError, match="positive finite"):
        learning.losses.factory.build_training_loss(
            _loss_config(kind="huber", beta=math.nan),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="data-only"):
        learning.losses.factory.build_training_loss(
            _loss_config(physics_enabled=True),
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="four values"):
        learning.losses.factory.build_training_loss(
            _loss_config(channel_weights=[1.0, 1.0]),
            device=torch.device("cpu"),
        )
