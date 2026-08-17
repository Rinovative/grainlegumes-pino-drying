"""Durable input-dataset, mean, table, schedule, map, and panel contracts."""

# ruff: noqa: PLR2004, S101

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from src import common, domain, generation
from src.analysis import generation_inputs
from src.analysis.generation_inputs import generation_input_controls as input_controls
from src.analysis.ui import tables

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
    if profile_id == "transient_drying":
        campaign = yaml.safe_load(path.read_text(encoding="utf-8"))
        operations_path = path.parent / campaign["operations_config"]
        operations = yaml.safe_load(operations_path.read_text(encoding="utf-8"))
        operations["boundary_schedule"]["startup_ramp"]["enabled"] = startup_enabled
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
    field_means: list[float] = []
    for record in summary.records:
        scalar = generation_inputs.diagnostics.field_statistics(record).loc[
            "eps_bed",
            "mean",
        ]
        field_means.append(
            generation_inputs.diagnostics._finite_real_scalar(  # noqa: SLF001
                scalar,
                label="eps_bed mean",
            )
        )
    expected_field_mean = float(np.mean(np.asarray(field_means, dtype=np.float64)))
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
        (0.0, 0.5, 1.0),
    )
    np.testing.assert_array_equal(canonical, output)
    assert startup.duration_h == 0.5
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


@pytest.mark.parametrize(
    "value",
    [1, 1.5, np.int64(2), np.float64(2.5)],
)
def test_finite_real_scalar_accepts_python_and_numpy_reals(value: object) -> None:
    """Accept the finite real scalar families produced by pandas reductions."""
    assert generation_inputs.diagnostics._finite_real_scalar(  # noqa: SLF001
        value,
        label="test value",
    ) == pytest.approx(float(cast("Any", value)))


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param("1.0", id="string"),
        pytest.param(pd.Timestamp("2024-01-01"), id="timestamp"),
        pytest.param(np.asarray((1.0,)), id="array"),
        pytest.param(pd.Series((1.0,)), id="series"),
    ],
)
def test_finite_real_scalar_rejects_nonreal_or_structured_values(value: object) -> None:
    """Reject ambiguous, non-finite, and structured pandas values clearly."""
    with pytest.raises((TypeError, ValueError), match=r"finite|real scalar"):
        generation_inputs.diagnostics._finite_real_scalar(  # noqa: SLF001
            value,
            label="test value",
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


def test_shared_selection_synchronizes_controls_and_preserves_local_state(
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

    assert state.selection.case_a_key == dataset_catalog.case_options(family_key)[2][1]
    assert second.case_a.value == 3
    assert callbacks == [1, 1]
    assert first.scale_lock is not None
    assert second.scale_lock is not None
    first.scale_lock.value = True
    assert first.selected_comparison().lock_scale is True
    assert second.selected_comparison().lock_scale is False
    assert callbacks == [2, 1]
    callbacks[:] = [0, 0]

    first.case_b.value = 3
    assert callbacks == [1, 1]
    technical_key = generation_inputs.sources.dataset_key(
        next(
            dataset
            for dataset in dataset_catalog.datasets
            if dataset.profile_id == "transient_drying" and dataset.campaign_purpose == "technical_runtime_smoke"
        )
    )
    second.dataset_b.value = technical_key

    assert state.selection.dataset_b_key == technical_key
    assert first.dataset_b.value == technical_key
    assert first.case_b.value == second.case_b.value == 1
    assert callbacks == [2, 2]


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


def test_panel_public_surface_accepts_catalog_and_rejects_invalid_input(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Construct the public panel only from an admitted dataset catalog."""
    panel = generation_inputs.panel.build_generation_input_eda_panel(
        datasets=dataset_catalog,
    )

    assert isinstance(panel, widgets.Output)
    with pytest.raises(TypeError):
        generation_inputs.panel.build_generation_input_eda_panel(
            datasets=cast("Any", []),
        )


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
    namespace: dict[str, Any] = {}
    exec(compile(source, "generation_input_eda.ipynb", "exec"), namespace)  # noqa: S102

    after = {path.relative_to(storage): path.read_bytes() for path in storage.rglob("*") if path.is_file()}
    assert after == before
    assert isinstance(namespace["workspace"], generation_inputs.workspace.GenerationInputEDAWorkspace)
    assert capsys.readouterr().out.strip() == namespace["workspace"].summary_text
    expected_panel = namespace["workspace"].panel
    assert shown == ([expected_panel] if expected_panel is not None else [])


def _map_norm_for_values(figure: Figure, expected: np.ndarray) -> object:
    """Return the normalization attached to one semantically matched map."""
    matches = []
    for binding in generation_inputs.plots.layout.map_colorbar_bindings(figure):
        collection = binding.anchor_axis.collections[0]
        values = np.asarray(collection.get_array())
        if values.size == expected.size and np.allclose(
            values.reshape(expected.shape),
            expected,
        ):
            matches.append(collection.norm)
    assert matches
    return matches[0]


def _figure_has_line(
    figure: Figure,
    expected_x: np.ndarray,
    expected_y: np.ndarray,
) -> bool:
    """Return whether a figure contains one exact semantic data series."""
    return any(
        np.array_equal(np.asarray(line.get_xdata()), expected_x) and np.allclose(np.asarray(line.get_ydata()), expected_y)
        for axis in figure.axes
        for line in axis.lines
    )


def test_basic_spatial_preserves_fields_pressure_and_scale_semantics(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Render exact fields, differences, pressure support, and scale policy."""
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
        first_field = first.fields["eps_bed"]
        second_field = second.fields["eps_bed"]
        difference = second_field - first_field
        default_first_norm = _map_norm_for_values(default, first_field)
        default_second_norm = _map_norm_for_values(default, second_field)
        locked_first_norm = _map_norm_for_values(locked, first_field)
        locked_second_norm = _map_norm_for_values(locked, second_field)
        difference_norm = _map_norm_for_values(default, difference)

        assert default_first_norm is not default_second_norm
        assert locked_first_norm is locked_second_norm
        assert isinstance(difference_norm, TwoSlopeNorm)
        assert difference_norm.vcenter == 0.0
        assert difference_norm.vmin is not None
        assert difference_norm.vmax is not None
        assert difference_norm.vmin == -difference_norm.vmax

        first_x, first_pressure = generation_inputs.diagnostics.inlet_pressure_boundary(first)
        second_x, second_pressure = generation_inputs.diagnostics.inlet_pressure_boundary(second)
        assert _figure_has_line(default, first_x, first_pressure)
        assert _figure_has_line(default, second_x, second_pressure)
        mean_pressure = np.mean(np.stack((first_pressure, second_pressure)), axis=0)
        assert _figure_has_line(default, first_x, mean_pressure)
    finally:
        plt.close(default)
        plt.close(locked)


def test_permeability_and_moisture_plots_preserve_scientific_values(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Render each composite from exact fields and nonlinear RH summaries."""
    steady_first, steady_second = profile_records["steady_flow"]
    steady_mean = generation_inputs.diagnostics.build_dataset_diagnostics((steady_first, steady_second))
    transient_first, transient_second = profile_records["transient_drying"]
    transient_mean = generation_inputs.diagnostics.build_dataset_diagnostics((transient_first, transient_second))
    contracts = (
        (
            generation_inputs.plots.permeability.tensor_comparison(
                steady_first,
                steady_mean,
                steady_second,
                steady_mean,
                lock_scale=False,
            ),
            steady_first,
            steady_second,
            generation_inputs.plots.permeability.TENSOR_FIELDS,
        ),
        (
            generation_inputs.plots.permeability.derived_comparison(
                steady_first,
                steady_mean,
                steady_second,
                steady_mean,
                lock_scale=False,
            ),
            steady_first,
            steady_second,
            generation_inputs.plots.permeability.DERIVED_FIELDS,
        ),
        (
            generation_inputs.plots.moisture.moisture_comparison(
                transient_first,
                transient_mean,
                transient_second,
                transient_mean,
                lock_scale=False,
            ),
            transient_first,
            transient_second,
            generation_inputs.diagnostics.MOISTURE_FIELD_NAMES,
        ),
    )
    assert all(isinstance(figure, Figure) for figure, *_rest in contracts)
    try:
        for figure, first, second, quantities in contracts:
            assert isinstance(figure, Figure)
            for quantity in quantities:
                _map_norm_for_values(figure, first.fields[quantity])
                _map_norm_for_values(figure, second.fields[quantity])
                _map_norm_for_values(
                    figure,
                    second.fields[quantity] - first.fields[quantity],
                )

        relationship = generation_inputs.plots.moisture._case_relationship(  # noqa: SLF001
            transient_first
        )
        statistics = generation_inputs.diagnostics.field_statistics(transient_first).loc["phi_eq"]
        startup = generation_inputs.diagnostics.transient_evidence(transient_first)[3]
        assert relationship == pytest.approx(
            (
                statistics["q05"],
                statistics["median"],
                statistics["q95"],
                startup.variables["phi_in_bc"].start,
                startup.variables["phi_in_bc"].end,
            )
        )
        assert generation_inputs.plots.moisture._mean_relationship(  # noqa: SLF001
            transient_mean
        ) == pytest.approx(
            (
                transient_mean.field_summary_means[("phi_eq", "q05")],
                transient_mean.field_summary_means[("phi_eq", "median")],
                transient_mean.field_summary_means[("phi_eq", "q95")],
                transient_mean.boundary_means["phi_in_bc start"],
                transient_mean.boundary_means["phi_in_bc startup end"],
            )
        )
    finally:
        for figure, *_rest in contracts:
            if isinstance(figure, Figure):
                plt.close(figure)


def test_schedule_plot_preserves_semantic_windows_and_common_mean(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Plot exact operating and first-hour support without duplicating means."""
    first, second = profile_records["transient_drying"]
    assert first.schedule is not None
    assert first.startup is not None
    source_schedule = np.array(first.schedule, copy=True)
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    figure = generation_inputs.plots.boundaries.schedule_comparison(
        first,
        summary,
        second,
        summary,
        same_dataset=True,
    )
    try:
        case_a_label = f"Case {first.case.case_index} (A)"
        case_b_label = f"Case {second.case.case_index} (B)"
        mean_label = f"Dataset mean, n = {summary.case_count}"
        lines = [line for axis in figure.axes for line in axis.lines]
        labels = {str(line.get_label()) for line in lines}

        assert {case_a_label, case_b_label, mean_label}.issubset(labels)
        assert {label for label in labels if "mean" in label.lower()} == {mean_label}
        assert all(
            line.get_linestyle() == "-" and line.get_marker() in {"None", None, ""}
            for line in lines
            if line.get_label() in {case_a_label, case_b_label}
        )
        assert all(line.get_linestyle() == "--" for line in lines if line.get_label() == mean_label)

        startup_end_h = first.startup.duration_h
        persisted_times_h = source_schedule[:, 0]
        expected_operating_h = persisted_times_h[persisted_times_h >= startup_end_h]
        display_times_h = np.linspace(0.0, 1.0, 61, dtype=np.float64)
        expected_early_minutes = 60.0 * display_times_h
        observed_case_supports = [np.asarray(line.get_xdata(), dtype=np.float64) for line in lines if line.get_label() == case_a_label]
        assert observed_case_supports
        assert all(
            np.array_equal(support, expected_operating_h) or np.array_equal(support, expected_early_minutes) for support in observed_case_supports
        )
        assert any(np.array_equal(support, expected_operating_h) for support in observed_case_supports)
        assert any(np.array_equal(support, expected_early_minutes) for support in observed_case_supports)

        phi_axis = next(axis for axis in figure.axes if axis.get_ylabel() == "phi_in_bc [1]" and axis.get_xlabel() == "time [min]")
        case_phi = next(line for line in phi_axis.lines if line.get_label() == case_a_label)
        mean_phi = next(line for line in phi_axis.lines if line.get_label() == mean_label)
        expected_case = generation_inputs.diagnostics.case_boundary_schedule(first, display_times_h)
        expected_mean = np.mean(
            np.stack(tuple(generation_inputs.diagnostics.case_boundary_schedule(record, display_times_h)[:, 3] for record in summary.records)),
            axis=0,
        )
        np.testing.assert_allclose(
            np.asarray(case_phi.get_ydata(), dtype=np.float64),
            expected_case[:, 3],
        )
        np.testing.assert_allclose(
            np.asarray(mean_phi.get_ydata(), dtype=np.float64),
            expected_mean,
        )

        assert expected_operating_h[0] == startup_end_h
        assert not np.any(expected_operating_h == 0.0)
        assert expected_early_minutes[0] == 0.0
        assert expected_early_minutes[-1] == 60.0
        assert 60.0 * startup_end_h in expected_early_minutes
        assert {axis.get_title() for axis in figure.axes} >= {
            "Operating schedule: 30 min onward",
            "Startup and early operation: 0-60 min",
        }
        np.testing.assert_array_equal(first.schedule, source_schedule)
    finally:
        plt.close(figure)


def test_disabled_startup_plot_uses_hourly_operation_without_hidden_support(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Treat ramp-disabled schedules as operating from time zero."""
    config = _config(
        generation_config_factory,
        "transient_drying",
        startup_enabled=False,
    )
    records = []
    for case_index in (1, 2):
        bundle = generation.cases.case.generate_case_input_bundle(
            config,
            case_index,
            tmp_path / f"case_{case_index:04d}",
        )
        admitted = generation.cases.admission.admit_input_case(
            bundle.directory,
            source_id="disabled-startup-input",
            label="disabled startup diagnostic fixture",
            batch_storage_name=config.batch_storage_name,
            campaign_purpose=str(config.scientific_values["campaign_purpose"]),
        )
        records.append(generation_inputs.diagnostics.build_case_diagnostics(admitted))

    first, second = records
    summary = generation_inputs.diagnostics.build_dataset_diagnostics(records)
    source_schedule = np.array(generation_inputs.diagnostics.transient_evidence(first)[0], copy=True)
    assert first.startup is not None
    assert first.startup.enabled is False
    np.testing.assert_array_equal(source_schedule[:3, 0], (0.0, 1.0, 2.0))
    assert not np.any(source_schedule[:, 0] == 0.5)
    np.testing.assert_array_equal(
        generation_inputs.diagnostics.operating_schedule_rows(first),
        source_schedule,
    )

    figure = generation_inputs.plots.boundaries.schedule_comparison(
        first,
        summary,
        second,
        summary,
        same_dataset=True,
    )
    try:
        titles = {axis.get_title() for axis in figure.axes}
        assert "Operating schedule" in titles
        assert "Early operation: 0-60 min" in titles
        assert not any("startup" in title.lower() for title in titles)
        case_label = f"Case {first.case.case_index} (A)"
        supports = [np.asarray(line.get_xdata(), dtype=np.float64) for axis in figure.axes for line in axis.lines if line.get_label() == case_label]
        assert any(np.array_equal(support, source_schedule[:, 0]) for support in supports)
        assert any(np.array_equal(support, 60.0 * np.linspace(0.0, 1.0, 61)) for support in supports)
        np.testing.assert_array_equal(first.schedule, source_schedule)
    finally:
        plt.close(figure)


def test_temperature_display_and_schedule_windows_preserve_source_evidence(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Display absolute temperatures in Celsius without altering persisted evidence."""
    first, second = profile_records["transient_drying"]
    assert first.startup is not None
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    values = np.asarray((273.15, 293.15))
    original_values = np.array(values, copy=True)
    assert generation_inputs.diagnostics.display_unit("T_in_base", "K") == "°C"
    assert generation_inputs.diagnostics.display_unit("T_in_amp", "K") == "K"
    np.testing.assert_allclose(
        generation_inputs.diagnostics.display_value("T_in_base", values, "K"),
        (0.0, 20.0),
    )
    np.testing.assert_array_equal(values, original_values)

    source_schedule = np.array(first.schedule, copy=True)
    startup_rows = generation_inputs.diagnostics.startup_schedule_rows(first)
    operating_rows = generation_inputs.diagnostics.operating_schedule_rows(first)
    startup_minutes = generation_inputs.diagnostics.startup_schedule_minutes(first)
    duration_h = generation_inputs.diagnostics.transient_evidence(first)[3].duration_h
    assert startup_rows.flags.writeable is False
    assert operating_rows.flags.writeable is False
    assert startup_minutes.flags.writeable is False
    assert np.all((startup_rows[:, 0] >= 0.0) & (startup_rows[:, 0] <= duration_h))
    assert np.all(operating_rows[:, 0] >= duration_h)
    assert operating_rows[0, 0] == duration_h
    np.testing.assert_array_equal(first.startup.support_times_h, startup_rows[:, 0])
    np.testing.assert_allclose(startup_minutes[:, 0], 60.0 * startup_rows[:, 0])
    np.testing.assert_array_equal(first.schedule, source_schedule)

    parameters = generation_inputs.diagnostics.parameter_comparison_table(
        first,
        summary,
        second,
        summary,
    )
    boundaries = generation_inputs.diagnostics.boundary_comparison_table(
        first,
        summary,
        second,
        summary,
    )
    baseline = parameters.loc[("Drying", "Inlet schedule", "Temperature baseline [°C]"), "Case A"]
    amplitude = parameters.loc[("Drying", "Inlet schedule", "Temperature amplitude [K]"), "Case A"]
    startup_temperature = boundaries.loc[("Drying", "Inlet temperature", "Start [°C]"), "Case A"]
    startup_delta = boundaries.loc[("Drying", "Inlet temperature", "Startup change [K]"), "Case A"]
    assert baseline == pytest.approx(first.case.payload["sampled_values"]["T_in_base"] - 273.15)
    assert amplitude == first.case.payload["sampled_values"]["T_in_amp"]
    assert startup_temperature == pytest.approx(first.startup.variables["T_in_bc"].start - 273.15)
    assert startup_delta == first.startup.variables["T_in_bc"].delta


def test_schedule_plot_rejects_mismatched_startup_durations(
    profile_records: dict[
        str,
        tuple[
            generation_inputs.diagnostics.GenerationInputDiagnostics,
            generation_inputs.diagnostics.GenerationInputDiagnostics,
        ],
    ],
) -> None:
    """Reject comparisons whose persisted startup semantics disagree."""
    first, second = profile_records["transient_drying"]
    assert second.startup is not None
    summary = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    changed_second = replace(
        second,
        startup=replace(second.startup, duration_h=second.startup.duration_h + 0.1),
    )
    with pytest.raises(ValueError, match="different persisted startup policies"):
        generation_inputs.plots.boundaries.schedule_comparison(
            first,
            summary,
            changed_second,
            summary,
            same_dataset=True,
        )
