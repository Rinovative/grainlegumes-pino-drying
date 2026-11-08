"""Protect the physical steady-grid spacing handed to the relative H1 loss and metric.

Responsibilities:
  - Check that the spacing comes from the physical coordinate channels as (rows, columns) in metres
  - Check that an additional Evaluation case on another grid spacing is refused

Design principles:
  - Synthetic samples on a small anisotropic grid replace published Datasets

This module does NOT:
  - Verify the relative H1 algebra, coordinate-grid validation or the run lifecycle
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src import domain, experiments

_TASK = domain.tasks.registry.get_task("steady_flow")
_CONFIG = {"task": _TASK.id}
_SPACING_Y, _SPACING_X = 0.5, 0.25


def _loader(*, spacing_y: float = _SPACING_Y, spacing_x: float = _SPACING_X) -> SimpleNamespace:
    """Return a loader stand-in whose one sample carries physical coordinate channels on a 4 x 6 grid."""
    inputs = torch.zeros(_TASK.in_channels, 4, 6)
    inputs[_TASK.input_names.index("x")] = (torch.arange(6) * spacing_x)[None, :].expand(4, 6)
    inputs[_TASK.input_names.index("y")] = (torch.arange(4) * spacing_y)[:, None].expand(4, 6)
    return SimpleNamespace(dataset=[{"x": inputs}])


def test_spacing_is_rows_then_columns_in_metres() -> None:
    """Return the y spacing first and the x spacing second, also when additional Evaluation shares the grid."""
    spacing = experiments.run._steady_grid_spacing(_CONFIG, _loader(), _loader())

    assert spacing == pytest.approx((_SPACING_Y, _SPACING_X))


def test_additional_evaluation_on_another_spacing_is_refused() -> None:
    """Refuse an additional Evaluation case whose grid spacing differs from the training grid."""
    with pytest.raises(ValueError, match="training spacing"):
        experiments.run._steady_grid_spacing(_CONFIG, _loader(), _loader(spacing_x=2 * _SPACING_X))
