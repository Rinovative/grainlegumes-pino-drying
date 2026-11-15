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

from src import domain
from src.analysis.eda import eda_capabilities as capabilities
from src.analysis.eda.plots import eda_plot_case_statistics as case_statistics
from src.analysis.presentation import channel_semantics, display_labels, field_labels, moisture_display, visual_semantics


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


def test_unique_labels_keep_unique_bases_and_pick_facets_in_order() -> None:
    assert display_labels.unique_display_labels(["a", "b"], [("x",), ("y",)]) == ("a", "b")
    # The first facet is equal in the group, so the second one separates it.
    labels = display_labels.unique_display_labels(["m", "m"], [("p", "c1", "d"), ("p", "c2", "d")])
    assert labels == ("m · c1", "m · c2")
    # A None value disqualifies the facet for the whole group.
    labels = display_labels.unique_display_labels(["m", "m"], [(None, "c1"), ("p", "c2")])
    assert labels == ("m · c1", "m · c2")
    with pytest.raises(ValueError, match="No facet separates"):
        display_labels.unique_display_labels(["m", "m"], [("p",), ("p",)])


def test_unique_labels_show_only_tokens_that_differ_even_in_the_middle() -> None:
    first = "exp/fno_h64__lr-1__seed-9__ep-2__k-4__sch-cosine__b-a1__rollout-4"
    second = "exp/fno_h64__lr-1__seed-9__ep-2__k-4__sch-cosine__rev-wandb-pages__b-a1__rollout-4"
    labels = display_labels.unique_display_labels(["m", "m"], [(first,), (second,)])
    assert labels[1] == "m · rev-wandb-pages"
    labels = display_labels.unique_display_labels(["m", "m"], [("exp/a__seed-1__x",), ("exp/a__seed-2__x",)])
    assert labels == ("m · seed-1", "m · seed-2")


def test_unique_labels_give_the_item_without_distinguishing_token_the_bare_base() -> None:
    labels = display_labels.unique_display_labels(["m", "m"], [("exp/a__b",), ("exp/a__b__rev-x",)])
    assert labels == ("m", "m · rev-x")


def test_unique_labels_grow_the_ellipsis_until_unique() -> None:
    first, second = "A" * 15 + "1" + "B" * 25, "A" * 15 + "2" + "B" * 25
    labels = display_labels.unique_display_labels(["m", "m"], [(first,), (second,)])
    assert labels == (f"m · {'A' * 15}1…{'B' * 15}", f"m · {'A' * 15}2…{'B' * 15}")


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


def _cielab(color: str) -> np.ndarray:
    """Convert one sRGB hex color to CIELAB (D65)."""
    rgb = np.array([int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = matrix @ linear / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def test_fixed_palettes_are_distinct_and_pooled_role_colors_stand_apart_from_every_material() -> None:
    assert len(set(visual_semantics.MATERIAL_COLORS.values())) == len(visual_semantics.MATERIAL_COLORS)
    assert [spec.family for spec in visual_semantics.ARCHITECTURES] == ["fno", "uno", "rno", "pi-fno", "pi-uno"]
    assert len({spec.color for spec in visual_semantics.ARCHITECTURES}) == len(visual_semantics.ARCHITECTURES)
    assert visual_semantics.architecture_label("pi-uno") == "PI-U-NO"
    assert visual_semantics.material_color("lentil") == visual_semantics.series_style(material="lentil").color
    assert set(visual_semantics.MATERIAL_MARKERS) == set(visual_semantics.MATERIAL_COLORS)
    assert len(set(visual_semantics.ROLE_LINESTYLES.values())) == len(visual_semantics.ROLE_LINESTYLES)
    for role_color in visual_semantics.ROLE_COLORS.values():
        lab = _cielab(role_color)
        assert float(np.hypot(*lab[1:])) > 30.0  # not grey
        assert min(float(np.linalg.norm(lab - _cielab(material))) for material in visual_semantics.MATERIAL_COLORS.values()) >= 39.0


def test_series_style_takes_the_color_from_the_architecture_else_the_material_else_the_pooled_role() -> None:
    style = visual_semantics.series_style

    assert style(architecture="uno", material="chickpea", role="parameter_ood") == visual_semantics.SeriesStyle("#D55E00", "--", "s")
    assert style(material="kidney_bean", role="material_ood") == visual_semantics.SeriesStyle("#009E73", ":", None)
    assert style(role="id", pooled=True) == visual_semantics.SeriesStyle(visual_semantics.ROLE_COLORS["id"], "-", None)
    assert style(architecture="fno", role="id", pooled=True).color == visual_semantics.architecture_color("fno")
    assert style(material="lentil").linestyle == "-"


@pytest.mark.parametrize(
    "arguments",
    [
        {"architecture": "gru"},
        {"material": "quinoa"},
        {"material": "lentil", "role": "test"},
        {"role": "id", "pooled": True, "material": "lentil"},
        {"role": "id"},
        {"pooled": True},
    ],
)
def test_series_style_refuses_unknown_entities_and_a_missing_color_source(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"Unknown|pooled|needs"):
        visual_semantics.series_style(**arguments)  # type: ignore[arg-type]


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
    labels = {role: display_labels.material_role_display_label("red_lentil", role) for role in ("id", "ood", "material_ood")}
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
    assert capabilities.field_quantity_label(frame, "phi") == "φ [%]"
    assert capabilities.field_quantity_label(frame, "phi", control=True) == "φ [% RH]"
    assert capabilities.field_quantity_label(frame, "w_surf", mathtext=True) == r"$X_{\mathrm{wb,surf}}$ [%]"
    assert capabilities.field_quantity_label(frame, "w_surf", control=True) == "Surface moisture [% w.b.]"


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
            "parameters": {
                "varying_draw": 2.0,
                "single_dataset_only": single_dataset_value,
                "runtime_target_wet_fraction_limit": 0.04,
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

    celsius = field_labels.display_values(kelvin, "K", field="T")
    already_celsius = field_labels.display_values(celsius, "°C", field="T")
    difference = field_labels.display_values((2.5, -1.0), "K", field="T", quantity_kind="difference")

    np.testing.assert_allclose(celsius, (0.0, 26.85), rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(already_celsius, celsius, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(difference, (2.5, -1.0), rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(kelvin, original)
    assert field_labels.display_unit("K") == "°C"
    assert field_labels.display_unit("K", quantity_kind="difference") == "°C"


def test_relative_humidity_displays_in_percent_while_other_dimensionless_quantities_stay_unchanged() -> None:
    """Only a named humidity fraction becomes % RH (differences pp); unit "1" alone never selects percent."""
    stored = np.asarray((0.20, 0.01), dtype=np.float64)

    np.testing.assert_allclose(field_labels.display_values(stored[:1], "1", field="phi"), (20.0,))
    np.testing.assert_allclose(field_labels.display_values(stored[1:], "1", field="phi", quantity_kind="difference"), (1.0,))
    np.testing.assert_array_equal(field_labels.display_values(stored, "1", field="eps_bed"), stored)
    np.testing.assert_array_equal(stored, (0.20, 0.01))
    assert field_labels.display_unit("1", field="phi") == "%"
    assert field_labels.display_unit("1", field="phi", control=True) == "% RH"
    assert field_labels.display_unit("1", field="phi", quantity_kind="difference") == "pp"
    assert field_labels.display_unit("1", field="eps_bed") == field_labels.display_unit("1") == "-"
    with pytest.raises(ValueError, match="fraction"):
        field_labels.display_unit("%", field="phi")


def test_eda_water_shows_as_wet_basis_percent_with_bulk_means_over_the_case_density() -> None:
    """Each cell converts with its own dry bulk density; a mean is 100 Σw/(Σρ+Σw), never the mean of cell percentages."""
    water = np.asarray([[[100.0, 100.0]], [[50.0, 150.0]]])  # [time, y, x]
    density = np.asarray([[400.0, 900.0]])
    frame = pd.DataFrame(
        [{"state_trajectories": {"w_surf": water}, "static_fields": {"rho_bu_dry": density}, "time": {"regular_state_hours": [0.0, 1.0]}}]
    )
    frame.attrs.update(field_categories={"dynamic_state": ["w_surf"]}, field_units={"w_surf": "kg/m^3", "rho_bu_dry": "kg/m^3"})

    times, means, lower, upper = capabilities.field_display_trajectory(frame, frame.iloc[0], "w_surf")

    cells = 100.0 * water / (density + water)
    np.testing.assert_allclose(times, (0.0, 1.0))
    np.testing.assert_allclose(means, [100.0 * domain.moisture.bulk_wet_basis_moisture(state, density) for state in water])
    assert means[0] != pytest.approx(cells[0].mean())
    np.testing.assert_allclose((lower, upper), (cells.min(axis=(1, 2)), cells.max(axis=(1, 2))))
    assert capabilities.field_quantity_label(frame, "w_surf") == "X_wb,surf [%]"


def test_channel_controls_name_the_displayed_units() -> None:
    """Channel controls read the units the plots show: moisture in words and % w.b., phi in % RH, T in °C."""
    assert moisture_display.channel_label("w_surf", "w_surf", "kg/m^3") == "Surface moisture [% w.b.]"
    assert moisture_display.channel_label("phi", "φ", "1") == "φ [% RH]"
    assert moisture_display.channel_label("T", "T", "K") == "T [°C]"
    assert moisture_display.field_display_unit("w_gr", "kg/m^3", difference=True) == "pp"
