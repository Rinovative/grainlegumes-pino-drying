"""Durable input-dataset, mean, table, schedule, map, and panel contracts.

Responsibilities:
  - Check input catalogs, scientific means, tables, schedules, panels, and superseded renders
  - Check that campaign outcomes carry their sampled parameters into the failures-by-parameter figure

Design principles:
  - Selected dataset identity and exact support govern shared controls and aggregates

This module does NOT:
  - Generate production input campaigns or mutate authoritative fields
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import BoundaryNorm
from matplotlib.figure import Figure
from support.generation_fixtures import lentil_generation_config

from src import domain, generation
from src.analysis import generation_inputs
from src.analysis.generation_inputs import generation_input_controls as input_controls
from src.analysis.ui import tables

if TYPE_CHECKING:
    from collections.abc import Callable


def test_input_batch_merging_uses_all_unique_cases(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Merge bounded requests and make every canonical case contribute."""
    storage = tmp_path / "overlap-storage"
    config = lentil_generation_config(generation_config_factory, "transient_drying", storage_root=storage)
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
            generation_inputs.diagnostics._finite_real_scalar(
                scalar,
                label="eps_bed mean",
            )
        )
    expected_field_mean = float(np.mean(np.asarray(field_means, dtype=np.float64)))
    assert summary.field_summary_means[("eps_bed", "mean")] == pytest.approx(expected_field_mean)


def test_dataset_contexts_preserve_separate_scientific_means(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Keep distinct admitted datasets and their scientific means separate."""
    source_dataset, second_dataset = tuple(dataset for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying")
    source = dataset_catalog.dataset_diagnostics(generation_inputs.sources.dataset_key(source_dataset))
    second = dataset_catalog.dataset_diagnostics(generation_inputs.sources.dataset_key(second_dataset))
    assert source.batch_id != second.batch_id
    assert source.case_count == len(source.records)
    assert second.case_count == len(second.records)
    context = generation_inputs.diagnostics.case_context_table(
        source.records[0],
        source,
        second.records[0],
        second,
    )
    assert tuple(context.loc["campaign purpose"]) == (
        "source_generation",
        "source_generation",
    )
    assert tuple(context.loc["simulation profile"]) == (
        "transient_drying",
        "transient_drying",
    )
    assert tuple(context.loc["batch storage name"]) == (
        source.batch_storage_name,
        second.batch_storage_name,
    )


def test_corrupt_manifested_adapter_is_excluded_from_discovery(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Exclude a canonical source after one declared adapter changes."""
    storage = tmp_path / "conflict-storage"
    config = lentil_generation_config(generation_config_factory, "steady_flow", storage_root=storage)
    generation.cases.input_generation.generate_input_cases(
        config,
        2,
        storage_root=storage,
    )
    reference = generation.cases.admission.discover_input_batches(storage).sources[0].cases[0]
    fields_path = reference.case_directory / "fields.csv"
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
    assert startup.enabled is True
    expected_support = np.insert(canonical[:, 0], 1, startup.duration_h)
    np.testing.assert_array_equal(schedule[:, 0], expected_support)
    assert schedule.shape[1] == 3
    assert schedule[0, 0] == 0.0
    assert np.all(np.diff(schedule[:, 0]) > 0.0)
    np.testing.assert_array_equal(canonical, output)
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
    assert generation_inputs.diagnostics._finite_real_scalar(
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
        generation_inputs.diagnostics._finite_real_scalar(
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
    assert float(cast("Any", pressure_statistics["min"])) == pytest.approx(float(np.min(pressure_values)))
    assert float(cast("Any", pressure_statistics["mean"])) == pytest.approx(float(np.mean(pressure_values)))
    assert float(cast("Any", pressure_statistics["min"])) > 0.0
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
    source_key = generation_inputs.sources.dataset_key(
        next(dataset for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying" and len(dataset.cases) == 3)
    )
    resolution = generation_inputs.selection.resolve_generation_input_selection(
        dataset_catalog,
        preferred_dataset_key=source_key,
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

    assert state.selection.case_a_key == dataset_catalog.case_options(source_key)[2][1]
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
    second_key = next(
        key
        for key in (
            generation_inputs.sources.dataset_key(dataset) for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying"
        )
        if key != source_key
    )
    second.dataset_b.value = second_key

    assert state.selection.dataset_b_key == second_key
    assert first.dataset_b.value == second_key
    assert first.case_b.value == second.case_b.value == 1
    assert callbacks == [2, 2]


def test_selection_defaults_reconcile_sparse_and_single_case_datasets(
    dataset_catalog: generation_inputs.sources.GenerationInputDatasetCatalog,
) -> None:
    """Use canonical order and next admitted cases without dense assumptions."""
    source_dataset = next(dataset for dataset in dataset_catalog.datasets if dataset.profile_id == "transient_drying" and len(dataset.cases) == 3)
    source = source_dataset.publications[0]
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
        lambda *, storage_root, **_kwargs: catalog_with_issue if storage_root == tmp_path else pytest.fail("wrong storage root"),
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
        lambda *, storage_root, **_kwargs: catalog if storage_root == tmp_path else pytest.fail("wrong storage root"),
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
    assert isinstance(result.summary_text, str)
    assert result.summary_text


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


def test_notebook_executes_workspace_without_input_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the notebook workflow without generating canonical inputs."""
    shown: list[object] = []

    def reject_generation(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("notebook invoked canonical input generation")

    monkeypatch.setattr(
        generation.cases.input_generation,
        "run_campaign_input_generation",
        reject_generation,
    )
    monkeypatch.setattr("IPython.display.display", shown.append)

    notebook = json.loads(Path("notebooks/01_eda_generation_inputs.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")
    namespace: dict[str, Any] = {}
    exec(compile(source, "01_eda_generation_inputs.ipynb", "exec"), namespace)  # noqa: S102

    assert isinstance(namespace["workspace"], generation_inputs.workspace.GenerationInputEDAWorkspace)
    expected_panel = namespace["workspace"].panel
    assert expected_panel is None or expected_panel in shown


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
        assert isinstance(difference_norm, BoundaryNorm)
        assert len(difference_norm.boundaries) == 12
        assert difference_norm.Ncmap == 11
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

        relationship = generation_inputs.plots.moisture._case_relationship(transient_first)
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
        assert generation_inputs.plots.moisture._mean_relationship(transient_mean) == pytest.approx(
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
    """Plot exact operating and first-hour data without mutating the source."""
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
        startup_end_h = first.startup.duration_h
        persisted_times_h = source_schedule[:, 0]
        expected_operating_h = persisted_times_h[persisted_times_h >= startup_end_h]
        display_times_h = np.linspace(0.0, 1.0, 61, dtype=np.float64)
        expected_early_minutes = 60.0 * display_times_h
        observed_supports = [np.asarray(line.get_xdata(), dtype=np.float64) for axis in figure.axes for line in axis.lines]
        assert any(np.array_equal(support, expected_operating_h) for support in observed_supports)
        assert any(np.array_equal(support, expected_early_minutes) for support in observed_supports)

        phi_axis = next(axis for axis in figure.axes if axis.get_ylabel() == "phi_in_bc [1]" and axis.get_xlabel() == "time [min]")
        expected_first = generation_inputs.diagnostics.case_boundary_schedule(first, display_times_h)[:, 3]
        expected_second = generation_inputs.diagnostics.case_boundary_schedule(second, display_times_h)[:, 3]
        expected_mean = np.mean(
            np.stack(tuple(generation_inputs.diagnostics.case_boundary_schedule(record, display_times_h)[:, 3] for record in summary.records)),
            axis=0,
        )
        observed_phi = [np.asarray(line.get_ydata(), dtype=np.float64) for line in phi_axis.lines]
        for expected in (expected_first, expected_second, expected_mean):
            assert any(values.shape == expected.shape and np.allclose(values, expected) for values in observed_phi)

        assert expected_operating_h[0] == startup_end_h
        assert not np.any(expected_operating_h == 0.0)
        assert expected_early_minutes[0] == 0.0
        assert expected_early_minutes[-1] == 60.0
        assert 60.0 * startup_end_h in expected_early_minutes
        np.testing.assert_array_equal(first.schedule, source_schedule)
    finally:
        plt.close(figure)


def test_disabled_startup_plot_uses_regular_operation_without_hidden_support(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Treat a test-owned disabled ramp as operating from regular time zero."""
    config = lentil_generation_config(
        generation_config_factory,
        "transient_drying",
        startup_enabled=False,
        startup_duration_h=0.25,
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
    source_schedule, canonical, output, startup = generation_inputs.diagnostics.transient_evidence(first)
    source_schedule = np.array(source_schedule, copy=True)
    assert startup.enabled is False
    np.testing.assert_array_equal(source_schedule, canonical)
    np.testing.assert_array_equal(canonical, output)

    figure = generation_inputs.plots.boundaries.schedule_comparison(
        first,
        summary,
        second,
        summary,
        same_dataset=True,
    )
    try:
        display_times_h = np.linspace(0.0, 1.0, 61, dtype=np.float64)
        observed_supports = [np.asarray(line.get_xdata(), dtype=np.float64) for axis in figure.axes for line in axis.lines]
        assert any(np.array_equal(support, source_schedule[:, 0]) for support in observed_supports)
        assert any(np.array_equal(support, 60.0 * display_times_h) for support in observed_supports)
        phi_axis = next(axis for axis in figure.axes if axis.get_ylabel() == "phi_in_bc [1]" and axis.get_xlabel() == "time [min]")
        expected_first = generation_inputs.diagnostics.case_boundary_schedule(first, display_times_h)[:, 3]
        expected_second = generation_inputs.diagnostics.case_boundary_schedule(second, display_times_h)[:, 3]
        observed_phi = [np.asarray(line.get_ydata(), dtype=np.float64) for line in phi_axis.lines]
        for expected in (expected_first, expected_second):
            assert any(values.shape == expected.shape and np.allclose(values, expected) for values in observed_phi)
        np.testing.assert_array_equal(first.schedule, source_schedule)
    finally:
        plt.close(figure)


def test_temperature_display_preserves_source_evidence(
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


def test_candidate_outcome_rates_keep_attempts_pending_and_selection_separate() -> None:
    """A failed attempt is not a terminal failed candidate or a selected success."""
    from src.analysis.generation_inputs import generation_input_sources as sources  # noqa: PLC0415

    rows = tuple(
        {
            "task": "transient_drying",
            "campaign_name": "example",
            "source_name": "example__" + "a" * 16,
            "material_family": "lentil",
            "domain_profile": "reference_range",
            "case_id": f"case_{index:04d}",
            "origin": "primary",
            "design_key": "lhs",
            "outcome": outcome,
            "selected": outcome == "successful",
            "attempts": ({"class": "license"}, {"class": "technical"}),
            "failure": {"stage": "solver", "class": "numerical", "reason_code": "nonconvergence"} if outcome == "failed" else None,
            "parameters": {"kappa": 2.0 + index},
            "parameter_units": {"kappa": "m^2"},
        }
        for index, outcome in enumerate(("successful", "failed", "failed", "pending"), start=1)
    )
    catalog = sources.CampaignOutcomeCatalog(rows)
    failed = sources.query_campaign_outcomes(catalog, task="transient_drying", source_name="example__" + "a" * 16, outcome="failed")
    assert len(failed) == 2
    assert not any(row["selected"] for row in failed)
    rates = sources.campaign_outcome_rates(rows).iloc[0]
    assert rates["successful"] == 1
    assert rates["failed"] == 2
    assert rates["pending"] == 1
    assert rates["success_fraction_terminal"] == pytest.approx(1 / 3)
    reasons = sources.campaign_failure_rates(rows).iloc[0]
    assert reasons["reason_code"] == "nonconvergence"
    assert reasons["terminal_candidates"] == 3
    assert reasons["reason_fraction_terminal"] == pytest.approx(2 / 3)
    parameters = sources.outcome_parameter_values(rows, "kappa", "m^2")
    np.testing.assert_array_equal(parameters["failed"], (4.0, 5.0))
    assert sources.outcome_parameter_values(rows, "kappa", "1")["failed"].size == 0


def test_failures_by_parameter_places_terminal_candidates_by_material_and_typed_reason() -> None:
    """The outcome table carries every sampled parameter; the figure draws terminal candidates only, per material and reason."""
    from src.analysis.generation_inputs import generation_input_sources as sources  # noqa: PLC0415

    def row(material: str, case_id: str, outcome: str, reason: str | None, kappa: float) -> dict[str, Any]:
        failure = None if reason is None else {"stage": "solver", "class": "solver", "reason_code": reason}
        return {"material_family": material, "case_id": case_id, "outcome": outcome, "failure": failure, "parameters": {"kappa": kappa, "eps": 0.4}}

    rows = (
        row("lentil", "case_0001", "failed", "stalled", 1.0),
        row("lentil", "case_0002", "successful", None, 2.0),
        row("chickpea", "case_0001", "failed", "timed_out", 3.0),
        row("chickpea", "case_0002", "pending", None, 4.0),
    )
    table = sources.campaign_outcome_table(rows)
    assert sources.campaign_outcome_parameters(table) == ("kappa", "eps")
    assert table["kappa"].tolist() == [1.0, 2.0, 3.0, 4.0]
    figure = generation_inputs.plots.overview.failures_by_parameter(table, sources.campaign_outcome_parameters(table))
    try:
        panels = [axis for axis in figure.axes if axis.get_title()]
        assert [axis.get_title() for axis in panels] == ["kappa", "eps"]
        assert [label.get_text() for label in panels[0].get_yticklabels()] == ["chickpea", "lentil"]
        assert sorted(text.get_text() for text in figure.legends[0].get_texts()) == ["stalled (n=1)", "successful (n=1)", "timed_out (n=1)"]
        assert sorted(float(x) for collection in panels[0].collections for x, _y in np.asarray(collection.get_offsets())) == [1.0, 2.0, 3.0]
    finally:
        plt.close(figure)


def test_superseded_live_render_cannot_replace_latest_export(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer selection can finish while an older render is still returning."""
    from src.analysis.generation_inputs import generation_input_panel as panel  # noqa: PLC0415

    class LocalControls:
        widget = widgets.VBox()
        callback: Callable[[], None] = lambda: None

        def set_callback(self, callback: Callable[[], None]) -> None:
            self.callback = callback

    local = LocalControls()
    old_figure = plt.figure()
    latest_figure = plt.figure()
    exports = {"plot_name": "candidate_view", "fig": None}
    published = []
    monkeypatch.setattr(panel, "display", published.append)
    calls = 0

    def render() -> Figure:
        nonlocal calls
        calls += 1
        if calls == 1:
            local.callback()
            return old_figure
        return latest_figure

    view, _activate = panel._live_view(local, render, export_state=exports, export_plot_name="candidate_view", export_title="Candidate")
    try:
        assert exports["fig"] is latest_figure
        assert old_figure not in published
        assert latest_figure in published
    finally:
        view.close()
        local.widget.close()
        plt.close(old_figure)
        plt.close(latest_figure)


def test_campaign_outcomes_are_discovered_by_default(generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discover campaign outcome records without a separate admission mode."""
    storage = tmp_path / "storage"
    lentil_generation_config(generation_config_factory, "transient_drying", storage_root=storage)
    campaign_name = next(iter((storage / "01_generation" / "transient_drying").iterdir())).name
    rows = ({"source_name": campaign_name, "outcome": "successful"}, {"source_name": campaign_name, "outcome": "pending"})
    monkeypatch.setattr(generation.publication.catalog, "list_campaign_cases", lambda *_args, **_kwargs: rows)
    catalog = generation_inputs.sources.discover_generation_input_datasets(storage)
    assert tuple(row["outcome"] for row in catalog.outcomes.rows) == ("successful", "pending")
