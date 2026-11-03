"""Protect shared field labels, unit conversions, and visual normalization contracts.

Responsibilities:
  - Check label uniqueness, field ordering, dataset colors, statistic visibility, and display units

Design principles:
  - Display conversion preserves authoritative arrays and explicit experiment roles

This module does NOT:
  - Assign scientific roles or fit preprocessing scales
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_case_statistics as case_statistics
from src.analysis.presentation import channel_semantics, display_labels, field_labels, visual_semantics


def _label_metadata(*, identity: str, material: str = "red_lentil") -> display_labels.DatasetDisplayMetadata:
    return display_labels.DatasetDisplayMetadata(
        task_id="transient_drying",
        material_family=material,
        sampling_regime="natural",
        campaign_purpose="source_generation",
        domain_profile="reference_range",
        canonical_identity=identity,
    )


def test_source_labels_omit_sampling_regime_and_canonical_identity() -> None:
    metadata = _label_metadata(identity="batch__f91a8b")

    label = display_labels.dataset_display_label(metadata)
    assert "natural" not in label.casefold()
    assert "\n" not in label
    assert metadata.canonical_identity not in label


def test_colliding_labels_are_disambiguated_without_full_identity() -> None:
    common = {
        "task_id": "steady_flow",
        "material_family": "airflow_variation",
        "sampling_regime": "parameter_ood",
        "campaign_purpose": None,
        "domain_profile": "parameter_extension",
    }
    datasets = (
        display_labels.DatasetDisplayMetadata(**common, canonical_identity="alpha-001"),
        display_labels.DatasetDisplayMetadata(**common, canonical_identity="alpine-002"),
        _label_metadata(identity="other-003", material="chickpea"),
    )

    labels = display_labels.dataset_display_labels(datasets)
    assert len(set(labels)) == len(labels)
    assert labels[2] == display_labels.dataset_display_label(datasets[2])
    assert not any(dataset.canonical_identity in label for dataset, label in zip(datasets, labels, strict=True))


def test_channel_order_and_mixed_dataset_intersection_are_semantic() -> None:
    metadata = {
        "Kxx": channel_semantics.ChannelPresentationMetadata("airflow_input", order=0),
        "p": channel_semantics.ChannelPresentationMetadata("airflow_output", order=0),
        "U": channel_semantics.ChannelPresentationMetadata("airflow_output", order=4),
        "X_0_db_field": channel_semantics.ChannelPresentationMetadata("transient_input", order=0),
        "T": channel_semantics.ChannelPresentationMetadata("transient_output", order=0),
        "w_int": channel_semantics.ChannelPresentationMetadata("transient_output", order=3),
        "quality": channel_semantics.ChannelPresentationMetadata("diagnostic", order=1),
    }

    assert channel_semantics.ordered_channels(
        ("quality", "T", "x", "w_int", "p", "X_0_db_field", "U", "Kxx"),
        metadata=metadata,
    ) == ("Kxx", "p", "U", "X_0_db_field", "T", "w_int", "quality")


def test_visual_semantics_and_dataset_colors_are_deterministic() -> None:
    assert visual_semantics.field_visual_semantics("u").centered is True
    assert visual_semantics.field_visual_semantics("error", role="absolute_error").colormap == "Reds"
    assert visual_semantics.field_visual_semantics("error", role="signed_error").centered is True

    first = visual_semantics.DatasetVisualIdentity("identity-a", "Lentil · Natural")
    second = visual_semantics.DatasetVisualIdentity("identity-b", "Lentil · Natural")
    forward = visual_semantics.dataset_colors((first, second))
    reverse = visual_semantics.dataset_colors((second, first))

    assert forward == reverse
    assert set(forward) == {"identity-a", "identity-b"}
    assert forward["identity-a"] != forward["identity-b"]


def test_duplicate_canonical_dataset_identity_is_rejected() -> None:
    first = visual_semantics.DatasetVisualIdentity("identity-a", "One")
    second = visual_semantics.DatasetVisualIdentity("identity-a", "Two")

    with pytest.raises(ValueError, match="unique canonical identities"):
        visual_semantics.dataset_colors((first, second))


def test_meta_statistics_keep_all_discovered_fields_except_exact_denylist() -> None:
    discovered = (
        "varying_draw",
        "constant_setting",
        "nested_retained_scalar",
        "geometry_width",
        "geometry_height",
        "completion_target_wet_fraction_limit",
        "runtime_target_wet_fraction_limit",
        "r_surf_0",
        "nested_geometry_width",
        "nested_completion_target_wet_fraction_limit",
    )

    assert case_statistics._visible_metadata_statistic_keys(discovered) == [
        "varying_draw",
        "constant_setting",
        "nested_retained_scalar",
        "nested_geometry_width",
        "nested_completion_target_wet_fraction_limit",
    ]


def test_material_role_labels_are_distinct_per_role() -> None:
    labels = {role: display_labels.material_role_display_label("red_lentil", role) for role in ("id", "ood", "additional_evaluation")}
    assert len(set(labels.values())) == len(labels)
    assert all("lentil" in label.casefold() for label in labels.values())


def test_field_labels_use_authoritative_display_units() -> None:
    frame = pd.DataFrame([{}])
    frame.attrs["field_units"] = {
        "Kxx": "m^2",
        "p_in_bc": "Pa",
        "phi": "1",
        "w_surf": "kg/m^3",
    }
    frame.attrs["field_representations"] = {
        "Kxx": "dimensionless_log10_ratio_to_1_m2",
        "p_in_bc": "identity",
        "phi": "absolute_physical_state",
        "w_surf": "absolute_physical_state",
    }

    assert capabilities.field_quantity_label(frame, "Kxx").endswith("[-]")
    assert capabilities.field_quantity_label(frame, "p_in_bc").endswith("[Pa]")
    assert capabilities.field_quantity_label(frame, "phi").endswith("[-]")
    assert capabilities.field_quantity_label(frame, "w_surf").endswith("[kg/m^3]")
    assert capabilities.field_quantity_label(frame, "w_surf", mathtext=True).endswith("[kg/m^3]")


def test_meta_statistics_use_parameter_union_without_fit_or_variation_filters() -> None:
    constant_setting = 7.0
    single_dataset_value = 11.0
    first = case_statistics._case_parameter_values(
        {
            "parameters": {
                "varying_draw": 1.0,
                "constant_setting": constant_setting,
                "geometry_width": 0.4,
                "completion_target_wet_fraction_limit": 0.05,
            }
        }
    )
    second = case_statistics._case_parameter_values(
        {
            "generator": {
                "legacy": {
                    "parameters": {
                        "varying_draw": 2.0,
                        "single_dataset_only": single_dataset_value,
                        "runtime_target_wet_fraction_limit": 0.04,
                    }
                }
            }
        }
    )
    union = list(dict.fromkeys((*first, *second)))

    assert case_statistics._visible_metadata_statistic_keys(union) == [
        "varying_draw",
        "constant_setting",
        "single_dataset_only",
    ]
    assert first["constant_setting"] == constant_setting
    assert "single_dataset_only" not in first
    assert second["single_dataset_only"] == single_dataset_value


def test_generated_parameters_and_material_scalars_are_disjoint_semantic_roles() -> None:
    sampled = case_statistics._case_parameter_values(
        {
            "parameters": {
                "T_init": 300.0,
                "case_only_parameter": 4.0,
                "r_surf_0": 0.25,
            },
            "parameter_units": {
                "T_init": "K",
                "case_only_parameter": "1",
                "r_surf_0": "m",
            },
        }
    )
    generated_keys = set(case_statistics._visible_metadata_statistic_keys(tuple(sampled)))
    scalar_keys = set(case_statistics._scalar_material_parameter_keys(("r_surf_0", "f_surf", "not_a_material_scalar")))

    assert generated_keys == {"T_init", "case_only_parameter"}
    assert scalar_keys == {"r_surf_0", "f_surf"}
    assert generated_keys.isdisjoint(scalar_keys)
    assert case_statistics._case_parameter_units(
        {
            "parameter_units": {
                "T_init": "K",
                "case_only_parameter": "1",
            }
        }
    ) == {"T_init": "K", "case_only_parameter": "1"}


def test_temperature_display_conversion_preserves_source_and_difference_values() -> None:
    kelvin = np.asarray((273.15, 300.0), dtype=np.float64)
    original = kelvin.copy()

    celsius = field_labels.display_values(kelvin, "K")
    already_celsius = field_labels.display_values(celsius, "°C")
    difference = field_labels.display_values((2.5, -1.0), "K", quantity_kind="difference")

    np.testing.assert_allclose(celsius, (0.0, 26.85), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(already_celsius, celsius, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(difference, (2.5, -1.0), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(kelvin, original)
    assert field_labels.display_unit("K") == "°C"
    assert field_labels.display_unit("K", quantity_kind="difference") == "°C"
