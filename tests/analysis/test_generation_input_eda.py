"""Durable input-dataset, mean, table, schedule, map, and panel contracts."""

# ruff: noqa: PLR2004, S101

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.collections import PathCollection
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib.figure import Figure

from src import common, domain, generation
from src.analysis import generation_inputs
from src.analysis.eda import eda_panel
from src.analysis.generation_inputs import generation_input_controls as input_controls
from src.analysis.generation_inputs import generation_input_panel as input_panel
from src.analysis.presentation import registry as completed_presentation
from src.analysis.ui import notebook as ui_notebook
from src.analysis.ui import tables

if TYPE_CHECKING:
    from collections.abc import Iterator

    from matplotlib.axes import Axes

pytest_plugins = ("tests.generation.conftest",)


def _config(
    generation_config_factory: Any,
    profile_id: str,
    *,
    startup_enabled: bool = True,
    campaign_purpose: str = "technical_runtime_smoke",
) -> Any:
    """Build one compact maintained-profile Generation configuration."""
    path, _template = generation_config_factory(
        simulation_profile=profile_id,
        natural_count=4,
        campaign_purpose=campaign_purpose,
    )
    if profile_id == "transient_drying" and not startup_enabled:
        campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
        operations_path = path.parent / campaign["operations_config"]
        operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
        operations["boundary_schedule"]["startup_ramp"]["enabled"] = False
        operations_path.write_text(
            yaml.safe_dump(operations, sort_keys=False),
            encoding="utf-8",
        )
    return generation.cases.config.load_generation_config(
        path,
        only_batch=generation.cases.config.build_batch_name(
            profile_id,
            "lentil",
            "natural",
        ),
    )


@pytest.fixture
def profile_records(
    generation_config_factory: Any,
    tmp_path: Path,
) -> dict[
    str,
    tuple[
        generation_inputs.diagnostics.GenerationInputDiagnostics,
        generation_inputs.diagnostics.GenerationInputDiagnostics,
    ],
]:
    """Generate and admit two compact canonical cases for each profile."""
    result = {}
    for profile_id in ("steady_flow", "transient_drying"):
        config = _config(generation_config_factory, profile_id)
        records = []
        for case_index in (1, 2):
            bundle = generation.cases.case.generate_case_input_bundle(
                config,
                case_index,
                tmp_path / profile_id / f"case_{case_index:04d}",
            )
            admitted = generation.cases.admission.admit_input_case(
                bundle.directory,
                source_id=f"input-{profile_id}",
                label="technical label is not a selector contract",
                batch_storage_name=config.batch_storage_name,
                campaign_purpose=str(config.scientific_values["campaign_purpose"]),
            )
            records.append(generation_inputs.diagnostics.build_case_diagnostics(admitted))
        result[profile_id] = (records[0], records[1])
    return result


@pytest.fixture
def dataset_catalog(
    generation_config_factory: Any,
    tmp_path: Path,
) -> generation_inputs.sources.GenerationInputDatasetCatalog:
    """Publish compact steady and two compatible transient datasets."""
    storage = tmp_path / "catalog-storage"
    generation.cases.input_generation.generate_input_cases(
        _config(generation_config_factory, "steady_flow"),
        3,
        storage_root=storage,
    )
    generation.cases.input_generation.generate_input_cases(
        _config(
            generation_config_factory,
            "transient_drying",
            campaign_purpose="family_generalization",
        ),
        3,
        storage_root=storage,
    )
    generation.cases.input_generation.generate_input_cases(
        _config(
            generation_config_factory,
            "transient_drying",
            campaign_purpose="technical_runtime_smoke",
        ),
        2,
        storage_root=storage,
    )
    return generation_inputs.sources.discover_generation_input_datasets(storage)


def _walk(widget: widgets.Widget) -> Iterator[widgets.Widget]:
    """Yield one public widget tree."""
    yield widget
    for child in getattr(widget, "children", ()):
        yield from _walk(child)


def _captured_panel(captured: list[object]) -> widgets.VBox:
    """Return the outer VBox containing the sole tab container."""
    matches = [item for item in captured if isinstance(item, widgets.VBox) and len(item.children) == 3 and isinstance(item.children[2], widgets.Tab)]
    assert matches
    return matches[-1]


def test_input_batch_merging_uses_all_unique_cases(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Merge bounded requests and make every canonical case contribute."""
    storage = tmp_path / "overlap-storage"
    config = _config(generation_config_factory, "transient_drying")
    generation.cases.input_generation.generate_input_cases(
        config,
        2,
        case_start=1,
        storage_root=storage,
    )
    generation.cases.input_generation.generate_input_cases(
        config,
        2,
        case_start=2,
        storage_root=storage,
    )
    discovery = generation.cases.admission.discover_input_batches(storage)
    catalog = generation_inputs.sources.GenerationInputDatasetCatalog(
        generation.cases.admission.InputSourceDiscovery(
            tuple(discovery.sources),
            (),
        )
    )

    assert len(catalog.datasets) == 1
    dataset = catalog.datasets[0]
    assert len(dataset.publications) == 1
    assert [reference.case_index for reference in dataset.cases] == [1, 2, 3]
    assert len({reference.case_input_id for reference in dataset.cases}) == 3
    assert all(source.source_kind == "input_generated" for source in dataset.publications)
    assert [label for label, _value in catalog.case_options(generation_inputs.sources.dataset_key(dataset))] == ["1", "2", "3"]
    dataset_label = catalog.dataset_options()[0][0]
    assert dataset_label == "Drying · Lentil · natural · trs"
    assert "input-" not in dataset_label
    assert "/" not in dataset_label

    summary = catalog.dataset_diagnostics(generation_inputs.sources.dataset_key(dataset))
    assert summary.case_count == 3
    expected_parameter = np.mean([float(record.case.payload["sampled_values"]["T_init"]) for record in summary.records])
    assert summary.parameter_means["T_init"] == pytest.approx(expected_parameter)
    expected_field_mean = np.mean([generation_inputs.diagnostics.field_statistics(record).loc["eps_bed", "mean"] for record in summary.records])
    assert summary.field_summary_means[("eps_bed", "mean")] == pytest.approx(expected_field_mean)


@pytest.mark.parametrize(
    ("campaign_purpose", "expected"),
    [
        ("family_generalization", "fg"),
        ("technical_runtime_smoke", "trs"),
        ("technical_smoke", "ts"),
        ("parameter_ood", "po"),
        ("extreme_family_ood", "efo"),
        ("pilot", "p"),
    ],
)
def test_campaign_purpose_abbreviation_is_mechanical(
    campaign_purpose: str,
    expected: str,
) -> None:
    """Use only lowercase initials from canonical purpose components."""
    assert expected == generation_inputs.labels.campaign_purpose_abbreviation(campaign_purpose)
    assert "td" not in generation_inputs.labels.campaign_purpose_abbreviation("technical_runtime_smoke")


@pytest.mark.parametrize(
    ("profile_id", "material_family", "campaign_purpose", "expected"),
    [
        ("steady_flow", "lentil", "technical_runtime_smoke", "Airflow · Lentil · natural · trs"),
        ("transient_drying", "lentil", "technical_runtime_smoke", "Drying · Lentil · natural · trs"),
        ("transient_drying", "lentil", "family_generalization", "Drying · Lentil · natural · fg"),
        ("transient_drying", "chickpea", "family_generalization", "Drying · Chickpea · natural · fg"),
        ("transient_drying", "sunflower_seed", "family_generalization", "Drying · Sunflower seed · natural · fg"),
    ],
)
def test_profile_qualified_dataset_labels_use_canonical_metadata(
    profile_id: str,
    material_family: str,
    campaign_purpose: str,
    expected: str,
) -> None:
    """Put the compact profile first without changing canonical metadata."""
    metadata = generation_inputs.labels.DatasetLabelMetadata(
        profile_id=profile_id,
        material_family=material_family,
        sampling_regime="natural",
        campaign_purpose=campaign_purpose,
        batch_identity="a" * 64,
    )
    assert generation_inputs.labels.dataset_display_label(metadata) == expected
    assert generation_inputs.labels.profile_display_label("steady_flow") == "Airflow"
    assert generation_inputs.labels.profile_display_label("transient_drying") == "Drying"


def test_dataset_labels_context_and_collisions_preserve_separate_means(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Keep fg/trs datasets separate and suffix only colliding abbreviations."""
    assert {label for label, _key in dataset_catalog.dataset_options()} == {
        "Airflow · Lentil · natural · trs",
        "Drying · Lentil · natural · fg",
        "Drying · Lentil · natural · trs",
    }
    transient_options = dataset_catalog.dataset_options(profile_ids=("transient_drying",))
    keys = dict(transient_options)
    family = dataset_catalog.dataset_diagnostics(keys["Drying · Lentil · natural · fg"])
    technical = dataset_catalog.dataset_diagnostics(keys["Drying · Lentil · natural · trs"])
    assert family.batch_id != technical.batch_id
    assert family.case_count == 3
    assert technical.case_count == 2
    context = generation_inputs.diagnostics.case_context_table(
        family.records[0],
        family,
        technical.records[0],
        technical,
    )
    assert tuple(context.loc["campaign purpose"]) == (
        "family_generalization",
        "technical_runtime_smoke",
    )
    assert tuple(context.loc["simulation profile"]) == (
        "transient_drying",
        "transient_drying",
    )
    assert tuple(context.loc["profile label"]) == ("Drying", "Drying")
    assert tuple(context.loc["batch storage name"]) == (
        family.batch_storage_name,
        technical.batch_storage_name,
    )

    sources = [dataset.publications[0] for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying"]
    purposes = ("technical_smoke", "transient_support")
    colliding_sources = tuple(
        replace(
            source,
            campaign_purpose=purpose,
            cases=tuple(replace(reference, campaign_purpose=purpose) for reference in source.cases),
        )
        for source, purpose in zip(sources, purposes, strict=True)
    )
    colliding = generation_inputs.sources.GenerationInputDatasetCatalog(generation.cases.admission.InputSourceDiscovery(colliding_sources, ()))
    collision_labels = [label for label, _key in colliding.dataset_options()]
    assert all(label.startswith("Drying · Lentil · natural · ts · ") for label in collision_labels)
    assert {label.rsplit(" · ", maxsplit=1)[-1] for label in collision_labels} == {dataset.batch_identity[:8] for dataset in colliding.datasets}


def test_corrupt_manifested_adapter_is_excluded_from_discovery(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Exclude a canonical source after one declared adapter changes."""
    storage = tmp_path / "conflict-storage"
    config = _config(generation_config_factory, "steady_flow")
    generated = generation.cases.input_generation.generate_input_cases(
        config,
        2,
        storage_root=storage,
    )
    fields_path = generated.raw_directory / config.case_id(1) / "inputs" / "fields.csv"
    fields_path.write_text(
        fields_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    discovery = generation.cases.admission.discover_input_batches(storage)
    catalog = generation_inputs.sources.GenerationInputDatasetCatalog(discovery)

    assert discovery.sources == ()
    assert len(discovery.issues) == 1
    assert "adapter hash or size" in discovery.issues[0].message
    assert catalog.datasets == ()


def test_case_diagnostics_preserve_canonical_permeability_and_sorption(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Keep canonical per-case nonlinear derivations and exact schedules."""
    steady = profile_records["steady_flow"][0]
    transient = profile_records["transient_drying"][0]
    for record in (steady, transient):
        principal = domain.permeability.symmetric_tensor_diagnostics(
            record.fields["Kxx"],
            record.fields["Kxy"],
            record.fields["Kyy"],
        )
        np.testing.assert_allclose(
            record.fields["K_min"],
            principal.minimum_principal,
        )
        np.testing.assert_allclose(
            record.fields["K_max"],
            principal.maximum_principal,
        )
        np.testing.assert_allclose(
            record.fields["K_anisotropy"],
            principal.anisotropy_ratio,
        )
        assert record.fields["K_min"].flags.writeable is False

    assert steady.schedule is None
    schedule, canonical, output, startup = generation_inputs.diagnostics.transient_evidence(transient)
    np.testing.assert_array_equal(
        schedule[:3, 0],
        (0.0, 1.0 / 6.0, 1.0),
    )
    np.testing.assert_array_equal(canonical, output)
    assert startup.duration_h == 1.0 / 6.0
    expected_phi = domain.moisture.oswin_equilibrium_relative_humidity(
        transient.fields["X_0_db_field"],
        transient.case.payload["sampled_values"]["T_init"],
        a_osw=transient.scalars["A_osw"],
        b_osw=transient.scalars["B_osw"],
        c_osw=transient.scalars["C_osw"],
    )
    np.testing.assert_allclose(transient.fields["phi_eq"], expected_phi)


def test_dataset_schedule_mean_requires_exact_persisted_support(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Average channels directly and never resample differing supports."""
    first, second = profile_records["transient_drying"]
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    assert summary.schedule_mean is not None
    assert first.schedule is not None
    assert second.schedule is not None
    np.testing.assert_array_equal(
        summary.schedule_mean[:, 0],
        first.schedule[:, 0],
    )
    np.testing.assert_allclose(
        summary.schedule_mean[:, 1:],
        np.mean(
            np.stack((first.schedule[:, 1:], second.schedule[:, 1:])),
            axis=0,
        ),
    )

    changed = np.array(second.schedule, copy=True)
    changed[1, 0] += 1.0e-6
    incompatible = replace(second, schedule=changed)
    unavailable = generation_inputs.diagnostics.build_dataset_diagnostics((first, incompatible))
    assert unavailable.schedule_mean is None
    assert "supports differ exactly" in cast(
        "str",
        unavailable.schedule_mean_unavailable,
    )


def test_grouped_tables_expand_components_and_preserve_raw_values(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Keep exact four-value rows while grouping and expanding components."""
    first, second = profile_records["transient_drying"]
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    _pressure_x, pressure_values = generation_inputs.diagnostics.inlet_pressure_boundary(first)
    pressure_statistics = generation_inputs.diagnostics.field_statistics(first).loc["p_in_bc"]
    assert pressure_statistics["min"] == pytest.approx(float(np.min(pressure_values)))
    assert pressure_statistics["mean"] == pytest.approx(float(np.mean(pressure_values)))
    assert pressure_statistics["min"] > 0.0
    parameter_table = generation_inputs.diagnostics.parameter_comparison_table(
        first,
        summary,
        second,
        summary,
    )
    field_table = generation_inputs.diagnostics.field_summary_comparison_table(
        first,
        summary,
        second,
        summary,
    )
    boundary_table = generation_inputs.diagnostics.boundary_comparison_table(
        first,
        summary,
        second,
        summary,
    )
    expected_columns = ("Case A", "Mean A", "Case B", "Mean B")
    assert tuple(parameter_table.columns) == expected_columns
    assert tuple(field_table.columns) == expected_columns
    assert tuple(boundary_table.columns) == expected_columns
    for table in (parameter_table, field_table, boundary_table):
        assert isinstance(table.index, pd.MultiIndex)
        assert tuple(table.index.names[:2]) == ("Section", "Category")
        assert [name for name, _section in generation_inputs.diagnostics.grouped_table_sections(table)] == [
            "Airflow",
            "Drying",
        ]
        assert all("[" in label and label.endswith("]") for label in table.index.get_level_values(-1))

    drying_parameters = dict(generation_inputs.diagnostics.grouped_table_sections(parameter_table))["Drying"]
    component_labels = (
        "Smooth component weight [1]",
        "Event component weight [1]",
        "Trend component weight [1]",
    )
    component_rows = drying_parameters.loc[("Inlet schedule", list(component_labels)), :]
    assert tuple(component_rows.index.get_level_values("Parameter")) == component_labels
    np.testing.assert_allclose(
        component_rows.to_numpy(dtype=np.float64).sum(axis=0),
        np.ones(len(expected_columns)),
    )

    dataset_parameters = generation_inputs.diagnostics.dataset_parameter_table(summary)
    dataset_fields = generation_inputs.diagnostics.dataset_field_summary_table(summary)
    assert tuple(dataset_parameters.columns) == ("Case 1", "Case 2")
    assert tuple(dataset_fields.columns) == ("Case 1", "Case 2")
    assert "Case input identity" not in dataset_parameters.columns
    assert [name for name, _section in generation_inputs.diagnostics.grouped_table_sections(dataset_parameters)] == [
        "Airflow",
        "Drying",
    ]
    assert [name for name, _section in generation_inputs.diagnostics.grouped_table_sections(dataset_fields)] == [
        "Airflow",
        "Drying",
    ]
    assert not {
        "z-score",
        "standardized deviation",
        "B - A",
        "difference",
    }.intersection(parameter_table.columns)

    values = pd.DataFrame(
        {
            "Case A": [300.0, 0.4, 1.0e-5, 1.0e-9, 2.0, "alpha", True],
            "Mean A": [310.0, 0.5, 2.0e-5, 2.0e-9, 2.0, "alpha", False],
            "Case B": [320.0, 0.6, 3.0e-5, 3.0e-9, 2.0, "beta", True],
            "Mean B": [330.0, 0.7, 4.0e-5, 4.0e-9, 2.0, "beta", False],
        },
        index=(
            "temperature",
            "fraction",
            "rate",
            "permeability",
            "equal",
            "category",
            "boolean",
        ),
    )
    unchanged = values.copy(deep=True)
    colors = tables.row_local_color_matrix(values)
    styles = tables.row_local_style_matrix(values)
    pd.testing.assert_frame_equal(values, unchanged)
    assert tuple(styles.loc["temperature"]) == tuple(styles.loc["fraction"])
    assert tuple(styles.loc["temperature"]) == tuple(styles.loc["rate"])
    assert tuple(styles.loc["temperature"]) == tuple(styles.loc["permeability"])
    assert tuple(colors.loc["temperature"]) == tuple(colors.loc["fraction"])
    assert tuple(colors.loc["temperature"]) == tuple(colors.loc["rate"])
    assert tuple(colors.loc["temperature"]) == tuple(colors.loc["permeability"])
    assert all(isinstance(color, tables.TableCellColors) for color in colors.loc["temperature"])
    assert len(set(colors.loc["equal"])) == 1
    assert isinstance(colors.loc["equal"].iloc[0], tables.TableCellColors)
    assert not any(colors.loc["category"])
    assert not any(colors.loc["boolean"])

    rendered = tables.styled_dataframe(
        values.iloc[:5],
        title="Independent row colors",
        row_local=True,
    )
    html = cast("widgets.HTML", rendered.children[1]).value
    assert all(color.background in html for color in colors.iloc[:5].to_numpy().ravel() if isinstance(color, tables.TableCellColors))
    assert all(color.foreground in html for color in colors.iloc[:5].to_numpy().ravel() if isinstance(color, tables.TableCellColors))

    changed_other_row = values.copy(deep=True)
    changed_other_row.loc["rate"] *= 1000.0
    changed_colors = tables.row_local_color_matrix(changed_other_row)
    assert tuple(changed_colors.loc["temperature"]) == tuple(colors.loc["temperature"])


def test_shared_selection_synchronizes_created_and_lazy_controls(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Synchronize canonical A/B values while preserving view-local state."""
    family_key = generation_inputs.sources.dataset_key(
        next(dataset for dataset in dataset_catalog.datasets if dataset.campaign_purpose == "family_generalization")
    )
    resolution = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=family_key,
        preferred_case_index=1,
    )
    state = generation_inputs.selection.GenerationInputSelectionState(
        dataset_catalog,
        initial_selection=resolution.selection,
    )
    first = input_controls.PairCaseControls(
        dataset_catalog,
        selection_state=state,
        include_scale_lock=True,
    )
    second = input_controls.PairCaseControls(
        dataset_catalog,
        selection_state=state,
        include_scale_lock=True,
    )
    callbacks = [0, 0]
    first.set_callback(lambda: callbacks.__setitem__(0, callbacks[0] + 1))
    second.set_callback(lambda: callbacks.__setitem__(1, callbacks[1] + 1))

    first.case_a.value = 3

    assert second.case_a.value == 3
    assert callbacks == [1, 1]
    assert first.scale_lock is not None
    assert second.scale_lock is not None
    first.scale_lock.value = True
    assert first.scale_lock.value is True
    assert second.scale_lock.value is False

    factory = input_panel._pair_factory(  # noqa: SLF001
        dataset_catalog,
        state,
        lambda _selection: widgets.HTML("cached"),
        include_scale_lock=True,
    )
    lazy = factory()
    lazy_controls = cast("widgets.VBox", lazy.children[0])
    lazy_case_a = cast(
        "widgets.BoundedIntText",
        lazy_controls.children[0].children[1],
    )
    assert lazy_case_a.value == 3
    assert factory() is lazy

    first.case_b.value = 3
    technical_key = generation_inputs.sources.dataset_key(
        next(
            dataset
            for dataset in dataset_catalog.datasets
            if dataset.profile_id == "transient_drying" and dataset.campaign_purpose == "technical_runtime_smoke"
        )
    )
    second.dataset_b.value = technical_key
    assert first.dataset_b.value == technical_key
    assert first.case_b.value == second.case_b.value == 1


def test_shared_selection_renders_only_active_cached_export_view(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Keep hidden synchronized views from replacing active PDF state."""
    family_key = generation_inputs.sources.dataset_key(
        next(dataset for dataset in dataset_catalog.datasets if dataset.campaign_purpose == "family_generalization")
    )
    selection = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=family_key,
    ).selection
    state = generation_inputs.selection.GenerationInputSelectionState(
        dataset_catalog,
        initial_selection=selection,
    )
    counts = {"first": 0, "second": 0}

    def render(name: str) -> Figure:
        counts[name] += 1
        figure, _axis = plt.subplots()
        return figure

    first_factory = input_panel._pair_factory(  # noqa: SLF001
        dataset_catalog,
        state,
        lambda _selection: render("first"),
    )
    second_factory = input_panel._pair_factory(  # noqa: SLF001
        dataset_catalog,
        state,
        lambda _selection: render("second"),
    )
    export_state: dict[str, object] = {}
    first_view = first_factory(
        export_state=export_state,
        export_plot_name="first_view",
        export_title="First view",
    )
    second_factory(
        export_state=export_state,
        export_plot_name="second_view",
        export_title="Second view",
    )
    first_factory(
        export_state=export_state,
        export_plot_name="first_view",
        export_title="First view",
    )
    assert counts == {"first": 2, "second": 1}

    first_controls = cast("widgets.VBox", first_view.children[0])
    first_case_a = cast(
        "widgets.BoundedIntText",
        first_controls.children[0].children[1],
    )
    first_case_a.value = 3

    assert counts == {"first": 3, "second": 1}
    assert export_state["plot_name"] == "first_view"
    assert export_state["title"] == "First view"
    assert isinstance(export_state["fig"], Figure)


def test_returning_to_tab_refreshes_selected_cached_view(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh one newly visible stale view without hidden redraw fan-out."""
    family_key = generation_inputs.sources.dataset_key(
        next(dataset for dataset in dataset_catalog.datasets if dataset.campaign_purpose == "family_generalization")
    )
    selection = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=family_key,
    ).selection
    state = generation_inputs.selection.GenerationInputSelectionState(
        dataset_catalog,
        initial_selection=selection,
    )
    sections = generation_inputs.presentation.sections_for_profiles(dataset_catalog.profiles)
    counts = {view.key: 0 for section in sections for view in section.plots}

    def render(key: str) -> widgets.HTML:
        counts[key] += 1
        return widgets.HTML(key)

    factories = {
        key: input_panel._pair_factory(  # noqa: SLF001
            dataset_catalog,
            state,
            lambda _selection, key=key: render(key),
        )
        for key in counts
    }
    monkeypatch.setattr(
        input_panel,
        "_view_factories",
        lambda _catalog, _state: factories,
    )
    monkeypatch.setattr(input_panel, "display", lambda _value: None)
    monkeypatch.setattr(input_panel, "clear_output", lambda **_kwargs: None)
    monkeypatch.setattr(ui_notebook, "display", lambda _value: None)

    shell = input_panel._GenerationInputPanelShell(  # noqa: SLF001
        dataset_catalog,
        title="Generation-input EDA",
        export_dir="",
        selection_state=state,
    )
    shell._show_panel()  # noqa: SLF001
    first_key = sections[0].plots[0].key
    second_key = sections[1].plots[0].key
    shell._tabs.selected_index = 1  # noqa: SLF001
    second_before = counts[second_key]
    state.select_case_a(dataset_catalog.case_options(family_key)[2][1])
    assert counts[second_key] == second_before + 1
    first_before = counts[first_key]

    shell._tabs.selected_index = 0  # noqa: SLF001

    assert counts[first_key] == first_before + 1
    assert shell._export_state["plot_name"] == first_key  # noqa: SLF001


def test_selection_defaults_reconcile_sparse_and_single_case_datasets(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Use canonical order and next admitted cases without dense assumptions."""
    family = next(dataset for dataset in dataset_catalog.datasets if dataset.campaign_purpose == "family_generalization")
    source = family.publications[0]
    sparse_source = replace(
        source,
        cases=(source.cases[0], source.cases[2]),
    )
    sparse_catalog = generation_inputs.sources.GenerationInputDatasetCatalog(
        generation.cases.admission.InputSourceDiscovery(
            (sparse_source,),
            (),
        )
    )
    sparse_key = generation_inputs.sources.dataset_key(sparse_catalog.datasets[0])
    resolved = generation_inputs.selection.resolve_generation_input_selection(
        sparse_catalog,
        preferred_dataset_key=sparse_key,
        preferred_case_index=1,
    )
    assert sparse_catalog.reference(resolved.selection.case_a_key).case_index == 1
    assert sparse_catalog.reference(resolved.selection.case_b_key).case_index == 3

    missing_case = generation_inputs.selection.resolve_generation_input_selection(
        sparse_catalog,
        preferred_dataset_key=sparse_key,
        preferred_case_index=2,
    )
    assert sparse_catalog.reference(missing_case.selection.case_a_key).case_index == 1
    assert len(missing_case.issues) == 1

    one_source = replace(source, cases=(source.cases[0],))
    one_catalog = generation_inputs.sources.GenerationInputDatasetCatalog(generation.cases.admission.InputSourceDiscovery((one_source,), ()))
    one = generation_inputs.selection.resolve_generation_input_selection(one_catalog).selection
    assert one.case_b_key == one.case_a_key

    missing_dataset = generation_inputs.selection.resolve_generation_input_selection(
        sparse_catalog,
        preferred_dataset_key=("transient_drying", "f" * 64),
    )
    assert missing_dataset.selection.dataset_a_key == sparse_key
    assert len(missing_dataset.issues) == 1

    same_profile_fallback = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=("transient_drying", "e" * 64),
    )
    expected_transient = next(
        generation_inputs.sources.dataset_key(dataset) for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying"
    )
    assert same_profile_fallback.selection.dataset_a_key == expected_transient


def test_workspace_composes_catalog_summary_naming_defaults_and_panel(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Compose admitted owners and initialize the panel from catalog order."""
    captured: dict[str, Any] = {}
    panel_widget = widgets.Output()
    issue = generation.cases.admission.InputDiscoveryIssue(
        "input-rejected",
        tmp_path / "rejected",
        "manifest source identity mismatch",
    )
    catalog_with_issue = generation_inputs.sources.GenerationInputDatasetCatalog(
        generation.cases.admission.InputSourceDiscovery(
            tuple(publication for dataset in dataset_catalog.datasets for publication in dataset.publications),
            (issue,),
        )
    )

    monkeypatch.setattr(
        generation_inputs.workspace.sources,
        "discover_generation_input_datasets",
        lambda *, storage_root: catalog_with_issue if storage_root == tmp_path else pytest.fail("wrong storage root"),
    )

    def build_panel(**kwargs: Any) -> widgets.Output:
        captured.update(kwargs)
        return panel_widget

    monkeypatch.setattr(
        generation_inputs.workspace.panel,
        "build_generation_input_eda_panel",
        build_panel,
    )

    result = generation_inputs.workspace.prepare_generation_input_eda_workspace(
        storage_root=tmp_path,
        title="Input workspace",
    )

    assert isinstance(result, generation_inputs.workspace.GenerationInputEDAWorkspace)
    assert not isinstance(result, widgets.Widget)
    assert isinstance(result.summary_text, str)
    assert result.summary_text
    assert result.panel is panel_widget
    assert str(tmp_path) in result.summary_text
    assert f"Canonical input datasets: {len(catalog_with_issue.datasets)}" in result.summary_text
    assert f"Manifested input cases: {sum(len(dataset.cases) for dataset in catalog_with_issue.datasets)}" in result.summary_text
    assert "Drying · Lentil · natural · fg" in result.summary_text
    assert "input-rejected: manifest source identity mismatch" in result.summary_text
    assert "Campaign-purpose abbreviations:" in result.summary_text
    assert "- fg = family_generalization" in result.summary_text
    assert "- trs = technical_runtime_smoke" in result.summary_text
    assert "- Airflow = steady_flow" in result.summary_text
    assert "- Drying = transient_drying" in result.summary_text

    state = cast(
        "generation_inputs.selection.GenerationInputSelectionState",
        captured["selection_state"],
    )
    first_dataset = catalog_with_issue.datasets[0]
    first_key = generation_inputs.sources.dataset_key(first_dataset)
    assert state.selection.dataset_a_key == first_key
    assert state.selection.dataset_b_key == first_key
    assert state.selection.case_a_key == catalog_with_issue.case_options(first_key)[0][1]
    assert state.selection.case_b_key == catalog_with_issue.case_options(first_key)[1][1]
    assert captured["datasets"] is catalog_with_issue
    assert captured["title"] == "Input workspace"


def test_workspace_empty_catalog_is_actionable_and_does_not_build_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Render read-only generic guidance for an empty admitted catalog."""
    issue = generation.cases.admission.InputDiscoveryIssue(
        "input-invalid",
        tmp_path / "invalid",
        "manifest identity mismatch",
    )
    catalog = generation_inputs.sources.GenerationInputDatasetCatalog(generation.cases.admission.InputSourceDiscovery((), (issue,)))
    monkeypatch.setattr(
        generation_inputs.workspace.sources,
        "discover_generation_input_datasets",
        lambda *, storage_root: catalog if storage_root == tmp_path else pytest.fail("wrong storage root"),
    )
    monkeypatch.setattr(
        generation_inputs.workspace.panel,
        "build_generation_input_eda_panel",
        lambda **_kwargs: pytest.fail("empty catalog constructed an interactive panel"),
    )

    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = generation_inputs.workspace.prepare_generation_input_eda_workspace(
        storage_root=tmp_path,
    )
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    assert after == before
    assert isinstance(result, generation_inputs.workspace.GenerationInputEDAWorkspace)
    assert not isinstance(result, widgets.Widget)
    assert result.panel is None
    assert "No manifested canonical input cases were admitted" in result.summary_text
    assert str(tmp_path) in result.summary_text
    assert "generate-input-cases" in result.summary_text
    assert "simulation_generation.md" in result.summary_text
    assert "Skipped invalid input batches: 1" in result.summary_text
    assert "input-invalid: manifest identity mismatch" in result.summary_text


def test_abbreviation_legend_combines_configured_and_discovered_purposes() -> None:
    """Derive deduplicated deterministic legend rows through one helper."""
    rows = generation_inputs.labels.campaign_purpose_legend_rows(
        ("technical_runtime_smoke", "family_generalization"),
        ("family_generalization",),
    )
    assert [(row.abbreviation, row.campaign_purpose) for row in rows] == [
        ("fg", "family_generalization"),
        ("trs", "technical_runtime_smoke"),
    ]
    assert generation_inputs.labels.profile_label_rows(("steady_flow", "transient_drying")) == (
        ("Airflow", "steady_flow"),
        ("Drying", "transient_drying"),
    )


def test_controls_panel_and_notebook_use_one_ab_first_model(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use numeric cases, compatible datasets, automatic render, and no refresh."""
    assert isinstance(
        generation_inputs.panel.build_generation_input_eda_panel(datasets=dataset_catalog),
        widgets.Output,
    )
    with pytest.raises(TypeError):
        generation_inputs.panel.build_generation_input_eda_panel(
            datasets=cast("Any", []),
        )
    initial = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=dataset_catalog.dataset_options(profile_ids=("transient_drying",))[0][1],
    ).selection
    selection_state = generation_inputs.selection.GenerationInputSelectionState(
        dataset_catalog,
        initial_selection=initial,
    )
    local = input_controls.PairCaseControls(
        dataset_catalog,
        selection_state=selection_state,
        include_scale_lock=True,
    )
    assert local.dataset_b.value == local.dataset_a.value
    assert isinstance(local.case_a, widgets.BoundedIntText)
    assert isinstance(local.case_b, widgets.BoundedIntText)
    assert local.case_a.value == 1
    assert local.case_b.value == 2
    assert (local.case_a.min, local.case_a.max) == (1, 3)
    assert (local.case_b.min, local.case_b.max) == (1, 3)
    view_section = ui_notebook.make_dropdown_section(
        [("1-3. Field summaries", lambda: None, "field_summaries")],
        select_first=True,
    )
    view_selector = cast("widgets.Dropdown", view_section.children[0])
    expected_view_width = f"{ui_notebook.COMPACT_VIEW_SELECTOR_WIDTH_PX}px"
    expected_dataset_width = f"{input_controls.DATASET_SELECTOR_WIDTH_PX}px"
    assert view_selector.layout.width == expected_view_width
    assert local.dataset_a.layout.width == expected_dataset_width
    assert local.dataset_b.layout.width == expected_dataset_width
    assert input_controls.DATASET_SELECTOR_WIDTH_PX > ui_notebook.COMPACT_VIEW_SELECTOR_WIDTH_PX
    assert local.dataset_a.style.description_width == (f"{input_controls.DATASET_LABEL_WIDTH_PX}px")
    assert local.dataset_b.style.description_width == (local.dataset_a.style.description_width)
    assert local.dataset_a.layout.flex == local.dataset_b.layout.flex
    first_row, second_row = local.widget.children[:2]
    assert isinstance(first_row, widgets.HBox)
    assert isinstance(second_row, widgets.HBox)
    assert first_row.layout.display == second_row.layout.display == "flex"
    assert first_row.layout.flex_flow == second_row.layout.flex_flow == "row wrap"
    assert first_row.layout.grid_gap == second_row.layout.grid_gap
    assert local.case_a.layout.width == local.case_b.layout.width == (f"{input_controls.CASE_SELECTOR_WIDTH_PX}px")
    assert round(84 * 0.75) == input_controls.CASE_VALUE_WIDTH_PX
    assert input_controls.CASE_SELECTOR_WIDTH_PX == (input_controls.CASE_LABEL_WIDTH_PX + input_controls.CASE_VALUE_WIDTH_PX)
    assert local.case_a.style.description_width == (f"{input_controls.CASE_LABEL_WIDTH_PX}px")
    assert not isinstance(local.case_a, widgets.Dropdown)
    assert not isinstance(local.case_b, widgets.Dropdown)
    assert local.previous_a.layout.width == local.following_a.layout.width
    assert local.previous_b.layout.width == local.following_b.layout.width
    assert local.previous_a.layout.width == (f"{input_controls.CASE_STEP_WIDTH_PX}px")
    local.following_a.click()
    assert local.case_a.value == 2
    local.previous_a.click()
    assert local.case_a.value == 1
    local.case_a.value = 3
    assert local.selected_comparison().case_a.case.case_index == 3
    assert local.scale_lock is not None
    assert local.scale_lock.value is False
    assert {dataset_catalog.dataset(cast("Any", value)).profile_id for _label, value in local.dataset_b.options} == {"transient_drying"}

    transient_options = dataset_catalog.dataset_options(profile_ids=("transient_drying",))
    assert len(transient_options) == 2
    local.dataset_b.value = transient_options[1][1]
    comparison = local.selected_comparison()
    assert comparison.case_a.profile_id == comparison.case_b.profile_id
    assert comparison.same_dataset is False

    captured: list[object] = []
    monkeypatch.setattr(input_panel, "display", captured.append)
    monkeypatch.setattr(ui_notebook, "display", captured.append)
    result = generation_inputs.panel.build_generation_input_eda_panel(
        datasets=dataset_catalog,
    )
    assert isinstance(result, widgets.Output)
    open_button = next(item for item in captured if isinstance(item, widgets.Button) and item.description.endswith(" - Open"))
    open_button.click()
    outer = _captured_panel(captured)
    header, _status, tabs = outer.children
    assert isinstance(header, widgets.HBox)
    assert [button.description for button in header.children] == [
        "Close",
        "Export PDF",
    ]
    assert isinstance(tabs, widgets.Tab)
    assert len(tabs.children) == 4
    live_view = next(
        item
        for item in reversed(captured)
        if isinstance(item, widgets.VBox) and len(item.children) == 2 and isinstance(item.children[1], widgets.Output)
    )
    descriptions = [str(cast("Any", item).description) for item in _walk(live_view.children[0]) if hasattr(item, "description")]
    assert descriptions == [
        "Dataset A:",
        "Case A:",
        "←",
        "→",
        "Dataset B:",
        "Case B:",
        "←",
        "→",
    ]
    assert not any(isinstance(item, widgets.Button) and item.description in {"Update", "Refresh sources"} for item in _walk(outer))
    close_button = cast("widgets.Button", header.children[0])
    close_button.click()
    open_button.click()
    assert _captured_panel(captured) is outer

    sections = generation_inputs.presentation.GENERATION_INPUT_SECTIONS
    assert tuple(section.key for section in sections) == (
        "case_comparison",
        "boundary_schedule_comparison",
        "spatial_comparison",
        "dataset_overview",
    )
    assert "input_cases" not in inspect_signature_parameters(eda_panel.build_eda_panel)
    assert tuple(section.key for section in completed_presentation.EDA_SECTIONS) == ("metadata_fields", "spectral_analysis")


def test_notebook_executes_read_only_workspace_over_current_storage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute normal notebook code without generation or storage mutation."""
    storage = tmp_path / "storage"
    storage.mkdir()
    sentinel = storage / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    before = {path.relative_to(storage): path.read_bytes() for path in storage.rglob("*") if path.is_file()}
    shown: list[object] = []

    def reject_generation(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("notebook invoked canonical input generation")

    monkeypatch.setattr(
        common.paths,
        "get_storage_root",
        lambda storage_root=None: storage if storage_root is None else Path(storage_root),
    )
    monkeypatch.setattr(
        generation.cases.input_generation,
        "run_campaign_input_generation",
        reject_generation,
    )
    monkeypatch.setattr("IPython.display.display", shown.append)

    notebook = json.loads(Path("notebooks/generation_input_eda.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")
    forbidden = (
        "INPUT_GENERATION_MODE",
        "INPUT_CAMPAIGN_CONFIG",
        "INPUT_ONLY_BATCH",
        "INPUT_ALL_BATCHES",
        "INPUT_ONLY_REGIME",
        "INPUT_CASE_START",
        "INPUT_CASE_COUNT",
        "INPUT_ALL_CASES",
        "run_campaign_input_generation",
        "CampaignInputGenerationRequest",
        "dry_run",
        '"execute"',
        '"skip"',
    )
    assert not any(token in source for token in forbidden)
    namespace: dict[str, Any] = {}
    exec(compile(source, "generation_input_eda.ipynb", "exec"), namespace)  # noqa: S102

    after = {path.relative_to(storage): path.read_bytes() for path in storage.rglob("*") if path.is_file()}
    assert after == before
    assert isinstance(namespace["workspace"], generation_inputs.workspace.GenerationInputEDAWorkspace)
    assert capsys.readouterr().out.strip() == namespace["workspace"].summary_text
    expected_panel = namespace["workspace"].panel
    assert shown == ([expected_panel] if expected_panel is not None else [])


def inspect_signature_parameters(function: Any) -> tuple[str, ...]:
    """Return public signature parameter names without source inspection."""
    return tuple(inspect.signature(function).parameters)


_GEOMETRY_TOLERANCE = 1.0e-9


def _assert_colorbar_geometry(figure: Figure, *, expected: int) -> None:
    """Require every axes-coupled colorbar to match its map-axis height."""
    figure.canvas.draw()
    bindings = generation_inputs.plots.layout.map_colorbar_bindings(figure)
    assert len(bindings) == expected
    for binding in bindings:
        map_box = binding.anchor_axis.get_position()
        colorbar_box = binding.colorbar.ax.get_position()
        assert abs(colorbar_box.y0 - map_box.y0) <= _GEOMETRY_TOLERANCE
        assert abs(colorbar_box.y1 - map_box.y1) <= _GEOMETRY_TOLERANCE
        assert abs(colorbar_box.height - map_box.height) <= _GEOMETRY_TOLERANCE
        assert len(binding.map_axes) == 1
    assert figure.get_size_inches()[0] <= generation_inputs.plots.layout.MAP_LAYOUT.notebook_width


def _legend_labels(axis: Axes) -> list[str]:
    """Return legend text without repeated optional accesses."""
    legend = axis.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def _assert_embedded_row_colors(
    axis: Axes,
    table: pd.DataFrame,
) -> None:
    """Require embedded value cells to use the canonical row-local colors."""
    artist = axis.tables[0]
    colors = tables.row_local_color_matrix(table)
    assert np.allclose(artist[(0, 0)].get_facecolor(), to_rgba("#e7edf3"))
    for row in range(len(table.index)):
        assert np.allclose(
            artist[(row + 1, 0)].get_facecolor(),
            to_rgba("#f5f7fa"),
        )
        for column in range(len(table.columns)):
            value = table.iloc[row, column]
            cell = artist[(row + 1, column + 1)]
            cell_colors = colors.iloc[row, column]
            assert isinstance(cell_colors, tables.TableCellColors)
            assert np.allclose(
                cell.get_facecolor(),
                to_rgba(cell_colors.background),
            )
            assert cell.get_text().get_color() == cell_colors.foreground
            assert cell.get_text().get_text() == f"{float(value):.4g}"


def test_basic_spatial_uses_large_maps_pressure_line_and_physical_difference(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Keep exact map scales while presenting inlet pressure as a line."""
    first, second = profile_records["steady_flow"]
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    default = generation_inputs.plots.spatial.basic_comparison(
        first,
        summary,
        second,
        summary,
        lock_scale=False,
    )
    locked = generation_inputs.plots.spatial.basic_comparison(
        first,
        summary,
        second,
        summary,
        lock_scale=True,
    )
    assert isinstance(default, Figure)
    assert isinstance(locked, Figure)
    try:
        for figure in (default, locked):
            _assert_colorbar_geometry(figure, expected=3)
            assert figure.get_size_inches()[0] == pytest.approx(3 * generation_inputs.plots.layout.MAP_LAYOUT.map_column_width)
            pressure_axes = [axis for axis in figure.axes if axis.get_title() == "Inlet pressure boundary"]
            assert len(pressure_axes) == 1
            assert len(pressure_axes[0].lines) == 3
            assert all("Case " in str(line.get_label()) for line in pressure_axes[0].lines if line.get_linestyle() == "-")
            distribution_axis = next(axis for axis in figure.axes if axis.get_title().endswith("distribution"))
            assert (
                distribution_axis.get_position().width
                > generation_inputs.plots.layout.map_colorbar_bindings(figure)[0].anchor_axis.get_position().width
            )
            summary_axes = [axis for axis in figure.axes if axis.tables]
            assert len(summary_axes) == 1
            assert summary_axes[0].get_title() == ""
            summary_table = generation_inputs.diagnostics.field_summary_comparison_table(
                first,
                summary,
                second,
                summary,
                quantities=("eps_bed", "p_in_bc"),
            )
            _assert_embedded_row_colors(summary_axes[0], summary_table)
            bindings = generation_inputs.plots.layout.map_colorbar_bindings(figure)
            assert {binding.label for binding in bindings} == {"[1]"}
            assert all("Bed porosity" in binding.anchor_axis.get_title() for binding in bindings)
        default_bindings = generation_inputs.plots.layout.map_colorbar_bindings(default)
        locked_bindings = generation_inputs.plots.layout.map_colorbar_bindings(locked)
        assert default_bindings[0].anchor_axis.collections[0].norm is not default_bindings[1].anchor_axis.collections[0].norm
        assert locked_bindings[0].anchor_axis.collections[0].norm is locked_bindings[1].anchor_axis.collections[0].norm
        difference_norm = default_bindings[2].anchor_axis.collections[0].norm
        assert isinstance(difference_norm, TwoSlopeNorm)
        assert difference_norm.vcenter == 0.0
        assert difference_norm.vmin is not None
        assert difference_norm.vmax is not None
        assert difference_norm.vmin == -difference_norm.vmax
        plotted_difference = np.asarray(default_bindings[2].anchor_axis.collections[0].get_array()).reshape(first.fields["eps_bed"].shape)
        np.testing.assert_allclose(
            plotted_difference,
            second.fields["eps_bed"] - first.fields["eps_bed"],
        )
        map_titles = tuple(binding.anchor_axis.get_title() for binding in default_bindings)
        assert map_titles[0] == (f"Case {first.case.case_index}\nBed porosity")
        assert map_titles[1] == (f"Case {second.case.case_index}\nBed porosity")
        assert map_titles[2] == (f"Case {second.case.case_index} - Case {first.case.case_index}\nBed porosity")
    finally:
        plt.close(default)
        plt.close(locked)


def test_permeability_and_moisture_composites_embed_lower_content(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Keep aligned colored summaries, unit labels, and clean RH phases."""
    steady_first, steady_second = profile_records["steady_flow"]
    steady_mean = generation_inputs.diagnostics.build_dataset_diagnostics((steady_first, steady_second))
    transient_first, transient_second = profile_records["transient_drying"]
    transient_mean = generation_inputs.diagnostics.build_dataset_diagnostics((transient_first, transient_second))
    tensor = generation_inputs.plots.permeability.tensor_comparison(
        steady_first,
        steady_mean,
        steady_second,
        steady_mean,
        lock_scale=False,
    )
    derived = generation_inputs.plots.permeability.derived_comparison(
        steady_first,
        steady_mean,
        steady_second,
        steady_mean,
        lock_scale=False,
    )
    moisture = generation_inputs.plots.moisture.moisture_comparison(
        transient_first,
        transient_mean,
        transient_second,
        transient_mean,
        lock_scale=False,
    )
    assert isinstance(tensor, Figure)
    assert isinstance(derived, Figure)
    assert isinstance(moisture, Figure)
    contracts = (
        (
            tensor,
            steady_first,
            steady_mean,
            steady_second,
            steady_mean,
            generation_inputs.plots.permeability.TENSOR_FIELDS,
        ),
        (
            derived,
            steady_first,
            steady_mean,
            steady_second,
            steady_mean,
            generation_inputs.plots.permeability.DERIVED_FIELDS,
        ),
        (
            moisture,
            transient_first,
            transient_mean,
            transient_second,
            transient_mean,
            generation_inputs.diagnostics.MOISTURE_FIELD_NAMES,
        ),
    )
    try:
        for figure, first, mean_a, second, mean_b, quantities in contracts:
            _assert_colorbar_geometry(figure, expected=3 * len(quantities))
            distribution_axes = [axis for axis in figure.axes if axis.get_title().endswith("distribution")]
            assert len(distribution_axes) == len(quantities)
            assert all(axis.lines for axis in distribution_axes)
            assert all(
                axis.get_position().width > generation_inputs.plots.layout.map_colorbar_bindings(figure)[0].anchor_axis.get_position().width
                for axis in distribution_axes
            )
            summary_axis = next(axis for axis in figure.axes if axis.tables)
            assert summary_axis.get_title() == ""
            summary_table = generation_inputs.diagnostics.field_summary_comparison_table(
                first,
                mean_a,
                second,
                mean_b,
                quantities=quantities,
            )
            _assert_embedded_row_colors(summary_axis, summary_table)
            bindings = generation_inputs.plots.layout.map_colorbar_bindings(figure)
            expected_labels = tuple(f"[{generation_inputs.diagnostics.FIELD_UNITS[quantity]}]" for quantity in quantities for _column in range(3))
            assert tuple(binding.label for binding in bindings) == expected_labels
            for offset, quantity in enumerate(quantities):
                field_label = generation_inputs.diagnostics.FIELD_LABELS[quantity]
                assert all(field_label in bindings[offset * 3 + column].anchor_axis.get_title() for column in range(3))

        relation = next(axis for axis in moisture.axes if axis.get_title() == "Inlet vs bed-equilibrium RH")
        markers = [collection for collection in relation.collections if isinstance(collection, PathCollection)]
        assert markers
        assert all(
            np.allclose(
                marker.get_sizes(),
                generation_inputs.plots.moisture.RELATION_MARKER_SIZE,
            )
            for marker in markers
        )
        assert all(np.asarray(marker.get_edgecolor()).size == 0 for marker in markers)
        phase_colors = {tuple(np.asarray(marker.get_facecolor(), dtype=np.float64).reshape(-1, 4)[0].tolist()) for marker in markers}
        assert len(phase_colors) == 3

        expected_legend_labels = [
            "Bed median (q05-q95 range)",
            "Inlet start",
            "Inlet startup end",
        ]
        legend_axis = next(axis for axis in moisture.axes if _legend_labels(axis) == expected_legend_labels)
        legend = legend_axis.get_legend()
        assert legend is not None
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert legend_labels == expected_legend_labels
        assert not any("Dataset" in label or "role" in label for label in legend_labels)

        moisture.canvas.draw()
        histogram_axes = [axis for axis in moisture.axes if axis.get_title().endswith("distribution")]
        summary_axis = next(axis for axis in moisture.axes if axis.tables)
        summary_box = summary_axis.get_position()
        first_histogram_box = histogram_axes[0].get_position()
        assert abs(summary_box.y0 - first_histogram_box.y0) <= _GEOMETRY_TOLERANCE
        assert abs(summary_box.y1 - first_histogram_box.y1) <= _GEOMETRY_TOLERANCE
        assert abs(summary_box.x1 - relation.get_position().x1) <= _GEOMETRY_TOLERANCE
        renderer = cast("Any", moisture.canvas).get_renderer()
        summary_artist_box = summary_axis.tables[0].get_window_extent(renderer).transformed(moisture.transFigure.inverted())
        assert abs(summary_artist_box.x0 - summary_box.x0) <= _GEOMETRY_TOLERANCE
        assert abs(summary_artist_box.x1 - summary_box.x1) <= _GEOMETRY_TOLERANCE
        assert abs(summary_artist_box.y0 - summary_box.y0) <= _GEOMETRY_TOLERANCE
        assert abs(summary_artist_box.y1 - summary_box.y1) <= _GEOMETRY_TOLERANCE
        assert relation.get_position().height < histogram_axes[1].get_position().height

        legend_box = legend.get_window_extent()
        relation_box = relation.get_window_extent()
        figure_box = moisture.bbox
        assert legend_box.y1 <= relation_box.y0
        assert legend_box.width >= 0.9 * relation_box.width
        assert legend_box.x0 >= figure_box.x0
        assert legend_box.x1 <= figure_box.x1
        assert legend_box.y0 >= figure_box.y0
        assert legend_box.y1 <= figure_box.y1
    finally:
        for figure, *_contract in contracts:
            plt.close(figure)


def test_schedule_plot_draws_wide_markerless_cases_and_common_mean_once(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Draw wide markerless full/startup schedules on exact supports."""
    first, second = profile_records["transient_drying"]
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    figure = generation_inputs.plots.boundaries.schedule_comparison(
        first,
        summary,
        second,
        summary,
        same_dataset=True,
    )
    try:
        assert len(figure.axes) == 6
        assert all(len(axis.lines) == 3 for axis in figure.axes)
        assert figure.get_size_inches()[0] == pytest.approx(2 * generation_inputs.plots.layout.MAP_LAYOUT.map_column_width)
        first_axis = figure.axes[0]
        labels = [str(line.get_label()) for line in first_axis.lines]
        assert labels[0] == f"Case {first.case.case_index} (A)"
        assert labels[1] == "Dataset mean, n = 2"
        assert labels[2] == f"Case {second.case.case_index} (B)"
        assert first_axis.lines[0].get_linestyle() == "-"
        assert first_axis.lines[1].get_linestyle() == "--"
        assert all(line.get_marker() in {"None", None, ""} for axis in figure.axes for line in axis.lines)
        assert first.schedule is not None
        np.testing.assert_array_equal(
            first_axis.lines[0].get_xdata(),
            first.schedule[:, 0],
        )
        startup_axis = figure.axes[1]
        assert np.max(startup_axis.lines[0].get_xdata()) <= 1.0
    finally:
        plt.close(figure)
