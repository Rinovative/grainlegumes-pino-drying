"""Protect the physical grid spacing handed to the H1 state loss and the relative H1 metric.

Responsibilities:
  - Check that the spacing comes from the physical coordinate channels as (rows, columns) in metres, for both tasks
  - Check that an Evaluation Dataset on another coordinate grid is refused from manifest metadata alone

Design principles:
  - Synthetic samples on a small anisotropic grid replace published Datasets

This module does NOT:
  - Verify the H1 algebra, coordinate-grid validation or the run lifecycle
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src import datasets, domain, experiments

_TASK = domain.tasks.registry.get_task("steady_flow")
_CONFIG = {"task": _TASK.id}
_SPACING_Y, _SPACING_X = 0.5, 0.25


def _coordinates(names: tuple[str, ...], *, spacing_y: float, spacing_x: float) -> torch.Tensor:
    """Return zero channels ``names`` on a 4 x 6 grid whose ``x`` and ``y`` channels hold physical coordinates."""
    values = torch.zeros(len(names), 4, 6)
    values[names.index("x")] = (torch.arange(6) * spacing_x)[None, :].expand(4, 6)
    values[names.index("y")] = (torch.arange(4) * spacing_y)[:, None].expand(4, 6)
    return values


def test_spacing_is_rows_then_columns_in_metres() -> None:
    """Return the y spacing first and the x spacing second."""
    loader = SimpleNamespace(dataset=[{"x": _coordinates(_TASK.input_names, spacing_y=_SPACING_Y, spacing_x=_SPACING_X)}])

    assert experiments.run._grid_spacing(_CONFIG, loader) == pytest.approx((_SPACING_Y, _SPACING_X))


def test_transient_spacing_comes_from_the_static_coordinate_channels() -> None:
    """Read the transient grid from the physical coordinates of the static conditioning."""
    names = tuple(field.name for field in datasets.contracts.transient.TRANSIENT_STEP_CONTRACT.static_spatial_conditioning)
    loader = SimpleNamespace(dataset=[{"static": _coordinates(names, spacing_y=_SPACING_Y, spacing_x=_SPACING_X)}])

    assert experiments.run._grid_spacing({"task": "transient_drying"}, loader) == pytest.approx((_SPACING_Y, _SPACING_X))


def test_an_evaluation_dataset_on_another_coordinate_grid_is_refused() -> None:
    """The Train grid spacing serves every case, so an Evaluation Dataset must share the Train coordinate axes."""
    identity = datasets.contracts.identity
    train = identity.ManifestBinding(
        dataset_id="train",
        dataset_digest="d" * 64,
        channel_contract_digest="c" * 64,
        case_ids=("a",),
        physical_sources={"a": ("a", "simulation_a")},
        materials={"a": "lentil"},
        design_kinds={"a": "train"},
        case_partition=None,
        grid_axes_sha256="1" * 64,
    )
    identity.require_train_grid(train, [replace(train, dataset_id="same_grid")])
    with pytest.raises(ValueError, match="does not share the coordinate grid"):
        identity.require_train_grid(train, [replace(train, dataset_id="other_grid", grid_axes_sha256="2" * 64)])
