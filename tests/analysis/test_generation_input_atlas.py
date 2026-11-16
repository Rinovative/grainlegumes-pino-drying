"""Admitted atlas support, full-grid scales, and lazy selector contracts.

Responsibilities:
  - Check admitted map support, tensor fields, cyclic orientation labelled at its wrap ends, state bands, and selector reuse
  - Check the 2 x 3 field order and that the atlas draws equal maps with bars as tall as their maps on the shared grid
  - Check that axis and colour-bar units and quantity names come from one display rule and one name table

Design principles:
  - Display sampling preserves full-grid scales and endpoints

This module does NOT:
  - Generate permeability fields or fabricate maps for constant quantities
"""

from dataclasses import replace
from types import MappingProxyType
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.analysis.generation_inputs import generation_input_controls as controls
from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.generation_inputs import generation_input_selection as selection
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs.plots import generation_input_plot_atlas as atlas_plots
from src.analysis.generation_inputs.plots import generation_input_plot_layout as layout
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_field_maps as field_maps
from src.analysis.presentation.analysis_field_maps import FieldMapSemantic
from src.analysis.ui import analysis_ui_figure_layout as figure_layout

type ProfileRecords = dict[str, tuple[diagnostics.GenerationInputDiagnostics, diagnostics.GenerationInputDiagnostics]]


def test_atlas_keeps_true_support_and_exact_tensor_fields(profile_records: ProfileRecords) -> None:
    """Retain domain arrays, signed cross permeability and physical scales only."""
    record = profile_records["transient_drying"][0]
    atlas = atlas_plots.prepare_field_atlas(record)
    assert {"Kxx", "Kxy", "Kyy", "K_min", "K_max", "eps_bed", "X_0_db_field", "phi_eq"} <= set(atlas.fields)
    assert not {"p_in_bc", "T_in_bc", "phi_in_bc", "z", "z_bg"} & set(atlas.fields)
    assert atlas.fields["Kxy"] is record.fields["Kxy"]
    assert atlas.scales["Kxy"].semantic is FieldMapSemantic.LINEAR_SIGNED
    assert atlas.scales["Kxx"].semantic is FieldMapSemantic.LOG_POSITIVE
    assert len(atlas.scales["Kxy"].boundaries) - 1 in {9, 11, 13}  # an odd count of round signed bands
    first_scale = atlas.scales["Kxy"]
    first = atlas_plots.field_atlas(atlas, ("Kxy",))
    second = atlas_plots.field_atlas(atlas, ("Kxx", "Kxy"))
    assert atlas.scales["Kxy"] is first_scale
    assert atlas_plots.field_atlas(atlas, ()) is None
    plt.close(first)
    plt.close(second)


def test_atlas_orders_fields_as_a_two_by_three_grid_and_draws_equal_maps_with_unit_bars(profile_records: ProfileRecords) -> None:
    """The six main fields lead in grid order; every map has the planned size and a bar as tall as itself; the empty cells close the grid."""
    record = profile_records["transient_drying"][0]
    atlas = atlas_plots.prepare_field_atlas(record)
    leading = tuple(name for name in ("Kxx", "Kxy", "eps_bed", "Kyy", "K_orientation", "X_0_db_field") if name in atlas.fields)
    assert leading[:4] == ("Kxx", "Kxy", "eps_bed", "Kyy")  # the principal direction is the only main field a tensor may lack
    assert tuple(atlas.fields)[: len(leading)] == leading
    figure = atlas_plots.field_atlas(atlas, leading[:4])
    assert figure is not None
    try:
        figure.canvas.draw()
        maps, slots = figure.axes[:6], figure.axes[6:]  # two rows of three cells; the last row holds one map
        width, height = figure.get_size_inches()
        assert figure.get_layout_engine() is None
        assert all(axis.get_title() for axis in maps[:4])
        assert [axis.get_visible() for axis in maps] == [True] * 4 + [False] * 2
        assert maps[1].get_xlabel() == maps[2].get_xlabel() == "x [m]"
        assert maps[1].xaxis.label.get_fontsize() == maps[2].xaxis.label.get_fontsize() == figure_layout.LABEL_PT
        assert str(record.case.case_index) in figure.get_suptitle()
        sizes = np.array([(axis.get_position().width * width, axis.get_position().height * height) for axis in maps[:4]])
        np.testing.assert_allclose(sizes, np.broadcast_to(sizes[0], sizes.shape), rtol=1e-3)
        for map_axis, slot in zip(maps[:4], slots[:4], strict=True):
            colorbar = map_axis.collections[0].colorbar
            assert colorbar is not None
            assert colorbar.ax is slot
            drawn = colorbar.outline.get_window_extent().height  # the drawn bar with its arrows
            assert drawn == pytest.approx(map_axis.get_window_extent().height, rel=1e-2)
            assert slot.get_ylabel().endswith("]")
    finally:
        plt.close(figure)


def test_axis_and_colour_bar_units_share_one_display_rule_and_every_quantity_is_named() -> None:
    """Dimensionless reads [-] and m^2 reads m² on an axis as on a colour bar; each quantity has one label."""
    for quantity in diagnostics.FIELD_UNITS:
        assert layout.quantity_label(quantity)
        assert layout.quantity_axis_label(quantity).endswith(layout.quantity_colorbar_label(quantity))
    assert layout.quantity_axis_label("eps_bed").endswith(" [-]")
    assert layout.quantity_axis_label("Kxx").endswith(" [m²]")
    assert layout.quantity_label("X_0_db_field") == field_labels.field_label("X_0_db_field", mathtext=True)
    assert layout.quantity_label("K_anisotropy") == diagnostics.field_name("K_anisotropy")


def test_constant_density_and_isotropic_direction_are_not_fabricated_maps(profile_records: ProfileRecords) -> None:
    """Expose constant density as metadata and omit undefined orientation."""
    record = profile_records["transient_drying"][0]
    shape = record.fields["x"].shape
    fields = {**record.fields, "rho_bu_dry": np.full(shape, 900.0), "Kxy": np.zeros(shape), "Kyy": record.fields["Kxx"]}
    atlas = atlas_plots.prepare_field_atlas(replace(record, fields=MappingProxyType(fields)))
    assert "rho_bu_dry" not in atlas.fields
    assert "K_orientation" not in atlas.fields
    assert any("900" in note and "metadata" in note for note in atlas.notes)
    assert any("isotropic" in note for note in atlas.notes)


def test_axial_orientation_wraps_on_one_fixed_cyclic_scale(profile_records: ProfileRecords) -> None:
    """Nearly identical axes across the angle seam receive nearby colors."""
    record = profile_records["transient_drying"][0]
    shape = record.fields["x"].shape
    cross = np.full(shape, 1.0e-12)
    cross.flat[0] = -1.0e-12
    fields = {**record.fields, "Kxx": np.full(shape, 1.0e-8), "Kyy": np.full(shape, 2.0e-8), "Kxy": cross}
    atlas = atlas_plots.prepare_field_atlas(replace(record, fields=MappingProxyType(fields)))
    scale = atlas.scales["K_orientation"]
    np.testing.assert_allclose(scale.boundaries, np.arange(-90.0, 91.0, 20.0))  # nine round bands, the middle one centred on 0
    assert scale.is_locked
    assert scale.display_transform == "principal_axis_modulo_180_degrees"
    colors = scale.colormap(scale.normalizer(np.array([-89.99, 89.99])))
    assert np.linalg.norm(colors[0] - colors[1]) < 0.05
    figure = atlas_plots.field_atlas(atlas, ("K_orientation",))
    assert figure is not None
    figure.canvas.draw()
    slot = figure.axes[1]  # the map's colour-bar slot follows the one map
    np.testing.assert_allclose(slot.get_yticks(), scale.boundaries)  # labelled at both wrap ends, never at the band centres
    assert [label.get_text() for label in slot.get_yticklabels()][-1] == "90"
    plt.close(figure)


def test_state_fields_get_round_state_bands_so_a_near_uniform_one_is_widened(profile_records: ProfileRecords) -> None:
    """A state such as the porosity takes the state bands of its range; a near-uniform one is widened to steps of 0.001."""
    near_uniform = layout.physical_norm("eps_bed", 0.4 + 2e-4 * np.linspace(0.0, 1.0, 50))
    assert near_uniform.semantic is FieldMapSemantic.LINEAR_POSITIVE
    np.testing.assert_allclose(np.diff(near_uniform.boundaries), 1e-3)
    record = profile_records["transient_drying"][0]
    porosity = layout.physical_norm("eps_bed", record.fields["eps_bed"])
    np.testing.assert_array_equal(
        porosity.boundaries, field_maps.state_boundaries(float(np.min(record.fields["eps_bed"])), float(np.max(record.fields["eps_bed"])))
    )
    assert layout.physical_norm("eps_bed", np.full(4, 0.4)).semantic is FieldMapSemantic.CONSTANT


def test_atlas_field_toggles_reuse_admitted_case_and_scale_state(dataset_catalog: sources.GenerationInputDatasetCatalog) -> None:
    """A field toggle keeps exact case evidence without loading unrelated data."""
    state = selection.GenerationInputSelectionState(dataset_catalog)
    with patch.object(dataset_catalog, "load", wraps=dataset_catalog.load) as load:
        local = controls.FieldAtlasControls(dataset_catalog, state)
        atlas = local.selected_atlas()
        assert local.fields.selected == tuple(atlas.fields)
        local.fields.boxes["Kxx"].value = False
        assert "Kxx" not in local.fields.selected
        assert local.selected_atlas() is atlas
        load.assert_called_once_with(state.selection.case_a_key)


def test_atlas_display_sampling_preserves_full_grid_scale_and_endpoints(profile_records: ProfileRecords) -> None:
    """Changing map sampling cannot change scientific extrema or field arrays."""
    record = profile_records["transient_drying"][0]
    atlas = atlas_plots.prepare_field_atlas(record)
    original = np.array(atlas.fields["Kxx"], copy=True)
    full = atlas_plots.field_atlas(atlas, ("Kxx",), display_spatial_stride=1)
    sampled = atlas_plots.field_atlas(atlas, ("Kxx",), display_spatial_stride=4)
    assert full is not None
    assert sampled is not None
    try:
        full_image = full.axes[0].collections[0]
        sampled_image = sampled.axes[0].collections[0]
        full_array = full_image.get_array()
        sampled_array = sampled_image.get_array()
        assert full_array is not None
        assert sampled_array is not None
        assert full_image.norm is sampled_image.norm is atlas.scales["Kxx"].normalizer
        assert sampled_array.size < full_array.size
        assert sampled_array.ravel()[0] == original[0, 0]
        assert sampled_array.ravel()[-1] == original[-1, -1]
        np.testing.assert_array_equal(atlas.fields["Kxx"], original)
    finally:
        plt.close(full)
        plt.close(sampled)
