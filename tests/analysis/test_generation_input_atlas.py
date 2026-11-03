"""Admitted atlas support, full-grid scales, and lazy selector contracts.

Responsibilities:
  - Check admitted map support, tensor fields, cyclic orientation, and selector reuse

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

from src.analysis.generation_inputs import generation_input_controls as controls
from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
from src.analysis.generation_inputs import generation_input_selection as selection
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs.plots import generation_input_plot_atlas as atlas_plots
from src.analysis.presentation.analysis_field_maps import FieldMapSemantic

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
    assert len(atlas.scales["Kxy"].boundaries) == 12
    first_scale = atlas.scales["Kxy"]
    first = atlas_plots.field_atlas(atlas, ("Kxy",))
    second = atlas_plots.field_atlas(atlas, ("Kxx", "Kxy"))
    assert atlas.scales["Kxy"] is first_scale
    assert atlas_plots.field_atlas(atlas, ()) is None
    plt.close(first)
    plt.close(second)


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
    np.testing.assert_allclose(scale.boundaries, np.linspace(-90.0, 90.0, 12))
    assert scale.is_locked
    assert scale.display_transform == "principal_axis_modulo_180_degrees"
    colors = scale.colormap(scale.normalizer(np.array([-89.99, 89.99])))
    assert np.linalg.norm(colors[0] - colors[1]) < 0.05


def test_atlas_field_toggles_reuse_admitted_case_and_scale_state(dataset_catalog: sources.GenerationInputDatasetCatalog) -> None:
    """A field toggle keeps exact case evidence without loading unrelated data."""
    state = selection.GenerationInputSelectionState(dataset_catalog)
    local = controls.FieldAtlasControls(dataset_catalog, state)
    with patch.object(dataset_catalog, "load", wraps=dataset_catalog.load) as load:
        atlas = local.selected_atlas()
        assert local.fields.selected == tuple(atlas.fields)
        group = local.fields.group
        assert group is not None
        group.boxes["Kxx"].value = False
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
