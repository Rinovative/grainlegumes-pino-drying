"""Protect the switchable plot label style: symbols, names, or name and symbol from one field table.

Responsibilities:
  - Check each style's field, unit and metric labels for state, moisture, difference and velocity quantities
  - Check that control texts keep their symbols and percent basis in every style and that an invalid style raises

Design principles:
  - Every test that selects a style restores the default through one fixture

This module does NOT:
  - Render plots or check layout, colour scales or numerical values
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.analysis.presentation import field_labels, moisture_display

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _restore_default_style() -> Iterator[None]:
    yield
    field_labels.set_label_style("symbol")


def _labels() -> dict[str, str]:
    """Return one representative label of each kind in the active style."""
    return {
        "T": f"{field_labels.field_label('T', mathtext=True)} [{field_labels.display_unit('K', field='T')}]",
        "phi": f"{field_labels.field_label('phi')} [{field_labels.display_unit('1', field='phi')}]",
        "moisture": f"{moisture_display.display_name('w_surf')} [{moisture_display.field_display_unit('w_surf', 'kg/m^3')}]",
        "water density": f"{field_labels.field_label('w_surf')} [kg/m³]",
        "difference": moisture_display.field_display_unit("w_gr", "kg/m^3", difference=True),
        "metric": f"{field_labels.metric_label('nRMSE', 'T')} [{field_labels.display_unit('1')}]",
        "u": field_labels.field_label("u"),
        "v": field_labels.field_label("v"),
    }


def test_default_style_is_symbol() -> None:
    assert field_labels.label_style() == "symbol"
    assert _labels() == {
        "T": "$T$ [°C]",
        "phi": "φ [%]",
        "moisture": "X_wb,surf [%]",
        "water density": "w_surf [kg/m³]",
        "difference": "pp",
        "metric": "nRMSE(T) [-]",
        "u": "u",
        "v": "v",
    }


def test_name_style_writes_names_and_the_percent_basis_in_the_unit() -> None:
    field_labels.set_label_style("name")

    assert field_labels.label_style() == "name"
    assert _labels() == {
        "T": "Temperature [°C]",
        "phi": "Relative humidity [% RH]",
        "moisture": "Surface moisture [% w.b.]",
        "water density": "Surface water content [kg/m³]",
        "difference": "pp",
        "metric": "Temperature nRMSE [-]",
        "u": "Horizontal velocity",
        "v": "Vertical velocity",
    }


def test_name_symbol_style_writes_both_and_the_basis_in_the_symbol() -> None:
    field_labels.set_label_style("name_symbol")

    assert _labels() == {
        "T": "Temperature $T$ [°C]",
        "phi": "Relative humidity φ [%]",
        "moisture": "Surface moisture X_wb,surf [%]",
        "water density": "Surface water content w_surf [kg/m³]",
        "difference": "pp",
        "metric": "Temperature nRMSE(T) [-]",
        "u": "Horizontal velocity u",
        "v": "Vertical velocity v",
    }


@pytest.mark.parametrize("style", ["symbol", "name", "name_symbol"])
def test_control_texts_ignore_the_style(style: field_labels.LabelStyle) -> None:
    field_labels.set_label_style(style)

    assert moisture_display.channel_label("w_gr", "w_gr", "kg/m^3") == "Grain moisture [% w.b.]"
    assert moisture_display.channel_label("phi", field_labels.field_label("phi", control=True), "1") == "φ [% RH]"
    assert moisture_display.display_name("w_surf", control=True) == "Surface moisture"
    assert field_labels.field_label("undeclared_field") == "Undeclared field"


@pytest.mark.parametrize("style", ["Symbol", "", "names", None])
def test_invalid_label_style_raises_and_keeps_the_active_style(style: object) -> None:
    field_labels.set_label_style("name")

    with pytest.raises(ValueError, match="Label style"):
        field_labels.set_label_style(style)  # type: ignore[arg-type]
    assert field_labels.label_style() == "name"
