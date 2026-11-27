"""Display contracts of the generation-input views: Dataset defaults, candidate rates, case context, keys and notes.

Responsibilities:
  - Check that the dataset overviews open on the panel's Dataset A and the campaign outcomes on the campaign its batch
    locator names, never on another Source sharing its case inputs; that a campaign without candidates says its evidence
    is unavailable; and that a local choice outlives a later change of Dataset A
  - Check that candidate rates group by material, declared role (else domain profile) and origin over the group's
    terminal candidates, are listed by material in the display order (a held-out material last) and then by role in the
    design order, under every filter, that sampled parameters head the candidate preview by their table row labels, and
    that failure codes read as display text while the evidence details keep them exact
  - Check the case context in display names with input directories from the discovery root, and dataset tables with
    cases as rows
  - Check that transient-only views follow the transient Datasets and campaigns of the catalog and say when no transient
    Dataset can be selected
  - Check the shared key and the case captions and colours of a comparison figure, uniform fields included, one power of
    ten per field row that every bar writes its labels in unless a map notes its exception, the three significant digits, one
    notation per row and figure unit style of the summary table, the unfilled distributions and uniform-field lines with Case B's dashed and one
    count label, and the observed-range notes of one-signed differences and of crowded anisotropy and direction maps

Design principles:
  - Small synthetic candidate rows and the shared compact cases; no production data

This module does NOT:
  - Check map geometry, scales or scientific values, which the generation-input EDA and atlas tests own
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from src import common, generation
from src.analysis import generation_inputs
from src.analysis.generation_inputs import generation_input_panel as panel
from src.analysis.generation_inputs import generation_input_presentation as presentation
from src.analysis.generation_inputs import generation_input_sources as sources
from src.analysis.generation_inputs.plots import generation_input_plot_layout as layout
from src.analysis.generation_inputs.plots import generation_input_plot_overview as overview
from src.analysis.presentation import analysis_field_labels as field_labels
from src.analysis.presentation import analysis_visual_semantics as visual_semantics

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from matplotlib.axes import Axes

type CaseRecord = generation_inputs.diagnostics.GenerationInputDiagnostics
type ProfileRecords = dict[str, tuple[CaseRecord, CaseRecord]]

_POWER = re.compile(r"10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+")


def _widgets(roots: Iterable[widgets.Widget]) -> Iterator[widgets.Widget]:
    """Yield ``roots`` and every widget nested in them."""
    for root in roots:
        yield root
        yield from _widgets(getattr(root, "children", ()))


def _candidate(task: str, name: str, case_input_id: str | None, **changes: Any) -> dict[str, Any]:
    """Return one terminal candidate row of campaign ``(task, name)`` with the fields the outcome views read."""
    return {
        "task": task,
        "source_name": name,
        "material_family": "lentil",
        "sampling_regime": "test",
        "domain_profile": "reference_range",
        "origin": "primary",
        "case_id": "case_0001",
        "design_key": "test:1",
        "outcome": "successful",
        "failure": None,
        "case_input_id": case_input_id,
        **changes,
    }


def _dataset_a_state(catalog: sources.GenerationInputDatasetCatalog) -> tuple[sources.DatasetKey, Any]:
    """Return a transient Dataset A that is not the catalog's first Dataset, and the panel selection state on it."""
    key = next(sources.dataset_key(dataset) for dataset in catalog.datasets[1:] if dataset.profile_id == "transient_drying")
    resolution = generation_inputs.selection.resolve_generation_input_selection(catalog, preferred_dataset_key=key)
    return key, generation_inputs.selection.GenerationInputSelectionState(catalog, initial_selection=resolution.selection)


def _slot_texts(view: Any) -> list[str]:
    """Return the HTML texts the plot slot of ``view`` shows."""
    return [widget.value for widget in _widgets((view.slot.widget,)) if isinstance(widget, widgets.HTML)]


def test_dataset_overviews_open_on_dataset_a_and_keep_a_local_choice(dataset_catalog: sources.GenerationInputDatasetCatalog) -> None:
    """Both registered Dataset views start on the panel's Dataset A, never the catalog's first; a local choice outlives a later change of A."""
    key, state = _dataset_a_state(dataset_catalog)
    labels = {dataset_key: label for label, dataset_key in dataset_catalog.dataset_options()}
    steady = next(sources.dataset_key(dataset) for dataset in dataset_catalog.datasets if dataset.profile_id == "steady_flow")
    other = next(
        sources.dataset_key(dataset)
        for dataset in dataset_catalog.datasets
        if dataset.profile_id == "transient_drying" and dataset.batch_identity != key[1]
    )
    count = dataset_catalog.dataset_diagnostics(key).case_count
    factories = panel._view_factories(dataset_catalog, state)
    try:
        views = {name: factories[name]() for name in ("dataset_cases_parameters", "dataset_field_summaries")}
        for view in views.values():
            assert any("Drying · " in text and f"— {count} unique cases" in text for text in _slot_texts(view))
            (choice,) = (
                widget for widget in _widgets((view.widget,)) if isinstance(widget, widgets.ToggleButton) and widget.description == labels[steady]
            )
            choice.value = True  # the view's own choice
        state.select_dataset_a(other)
        for name, view in views.items():
            assert factories[name]() is view
            assert not view.failure.shown, name
            assert any("Airflow · " in text for text in _slot_texts(view)), name
    finally:
        plt.close("all")


def test_campaign_outcomes_open_on_the_campaign_its_batch_locator_names(
    dataset_catalog: sources.GenerationInputDatasetCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another Source sharing Dataset A's case inputs never opens; without own candidates the view says so; a local choice stays."""
    key, state = _dataset_a_state(dataset_catalog)
    dataset = dataset_catalog.dataset(key)
    locator = common.paths.campaign_batch_parts(dataset.batch_storage_name)
    assert locator is not None
    own, earlier = (locator[0], locator[1]), (locator[0], "a-original")
    shared = dataset.cases[0].case_input_id
    original = _candidate(*earlier, shared)  # an earlier Source of the task, listed first, that holds the same case inputs
    try:
        for rows in ((original, _candidate(*own, shared)), (original,)):
            monkeypatch.setattr(dataset_catalog, "outcomes", sources.CampaignOutcomeCatalog(rows))
            factory = panel._view_factories(dataset_catalog, state)["campaign_outcomes"]
            view = factory()
            (campaign,) = (widget for widget in _widgets(view.rows) if isinstance(widget, widgets.Dropdown) and widget.description == "Campaign")
            assert campaign.value == own
            assert str(campaign.label).startswith("Drying · ")
            assert not view.failure.shown
            assert any("evidence is unavailable" in text for text in _slot_texts(view)) == (len(rows) == 1)
        campaign.value = earlier  # the view's own choice
        state.select_dataset_a(sources.dataset_key(dataset_catalog.datasets[0]))
        assert factory() is view
        assert campaign.value == earlier
        assert any("Candidate outcomes" in text for text in _slot_texts(view))
        assert any(text.startswith("<details>") for text in _slot_texts(view))  # the evidence choice took the first candidate again
    finally:
        plt.close("all")


def test_candidate_rates_group_by_material_declared_role_or_domain_profile_and_origin() -> None:
    """Roles and origins of one material stay apart, a row without a declared role groups by its domain profile, never by design."""
    failure = {"stage": "solver", "class": "numerical", "reason_code": "nonconvergence"}
    undeclared = _candidate("transient_drying", "c", None, design_key="test:5", outcome="failed", failure=failure)
    del undeclared["sampling_regime"]
    rows = (
        _candidate("transient_drying", "c", None, design_key="test:1"),
        _candidate("transient_drying", "c", None, design_key="test:2", outcome="failed", failure=failure),
        _candidate("transient_drying", "c", None, design_key="test:3", sampling_regime="train"),
        _candidate("transient_drying", "c", None, design_key="test:4", origin="replacement", outcome="pending"),
        _candidate("transient_drying", "c", None, design_key="test:6", material_family="chickpea"),
        undeclared,
    )
    rates = sources.campaign_outcome_rates(rows)
    assert {(row.material_family, row.sampling_regime, row.origin): (row.successful, row.failed, row.pending) for row in rates.itertuples()} == {
        ("chickpea", "test", "primary"): (1, 0, 0),
        ("lentil", "reference_range", "primary"): (0, 1, 0),  # no declared role: its domain profile
        ("lentil", "test", "primary"): (1, 1, 0),  # two designs, one group
        ("lentil", "test", "replacement"): (0, 0, 1),  # pending only
        ("lentil", "train", "primary"): (1, 0, 0),
    }
    fractions = rates.set_index(["material_family", "sampling_regime", "origin"])["success_fraction_terminal"]
    assert fractions[("lentil", "test", "primary")] == pytest.approx(1 / 2)
    assert pd.isna(fractions[("lentil", "test", "replacement")])  # no terminal candidate, so no rate
    reasons = sources.campaign_failure_rates(rows)
    assert {(row.sampling_regime, row.terminal_candidates, row.reason_fraction_terminal) for row in reasons.itertuples()} == {
        ("reference_range", 1, 1.0),
        ("test", 2, 0.5),
    }
    shown = panel._display_candidates(rates)
    assert {"Material", "Role", "Origin"} <= set(shown.columns)
    assert set(shown["Material"]) == {"Lentil", "Chickpea"}
    assert set(shown["Role"]) == {"Test", "Train", "Reference Range"}


def test_rate_and_failure_tables_list_materials_in_the_display_order_and_roles_in_the_design_order() -> None:
    """
    The shown tables list the ID materials, then the held-out one, each by display label; per material Train, Validation, Test.

    A material whose frozen plan holds test designs only is held out (Material OOD); a row without a declared role follows the
    design kinds under its domain profile. The view keeps the plan's material order when an origin filter, or a partial checkpoint,
    leaves an ID material test designs only.
    """
    failure = {"stage": "solver", "class": "solver", "reason_code": "timed_out"}
    designs = (
        ("kidney_bean", "test", "primary"),  # held out; sorted by key it would precede lentil
        ("kidney_bean", "test", "replacement"),
        ("lentil", "test", "replacement"),
        ("lentil", "parameter_ood", "primary"),
        ("lentil", "test", "primary"),
        ("lentil", "validation", "primary"),
        ("lentil", "train", "primary"),
        ("chickpea", "test", "primary"),
        ("chickpea", "validation", "primary"),
        ("chickpea", "train", "primary"),
    )
    changes = ({"material_family": material, "sampling_regime": role, "origin": origin} for material, role, origin in designs)
    rows = tuple(
        _candidate("transient_drying", "c", None, case_id=f"case_{index:04d}", outcome="failed", failure=failure, **change)
        for index, change in enumerate(changes, start=1)
    )
    undeclared = {key: value for key, value in rows[-1].items() if key != "sampling_regime"} | {"material_family": "lentil", "case_id": "case_0099"}
    rows = (*rows, undeclared)
    expected = [
        ("Chickpea", "Train", "Primary"),
        ("Chickpea", "Validation", "Primary"),
        ("Chickpea", "Test", "Primary"),
        ("Lentil", "Train", "Primary"),
        ("Lentil", "Validation", "Primary"),
        ("Lentil", "Test", "Primary"),
        ("Lentil", "Test", "Replacement"),
        ("Lentil", "Parameter OOD", "Primary"),
        ("Lentil", "Reference Range", "Primary"),  # no declared role: its domain profile, after the design kinds
        ("Kidney bean", "Test", "Primary"),
        ("Kidney bean", "Test", "Replacement"),
    ]
    plan = {"chickpea": {"train": 2, "validation": 1, "test": 1}, "kidney_bean": {"test": 2}, "lentil": {"train": 2, "validation": 1, "test": 1}}
    materials = sources.campaign_material_order(plan)
    assert materials == ("chickpea", "lentil", "kidney_bean")
    tests_only = tuple(row for row in rows if row.get("sampling_regime") == "test" and row["material_family"] != "chickpea")  # a partial checkpoint
    assert list(sources.campaign_outcome_rates(tests_only, materials=materials)["material_family"].drop_duplicates()) == ["lentil", "kidney_bean"]
    for table in (sources.campaign_outcome_rates(rows, materials=materials), sources.campaign_failure_rates(rows, materials=materials)):
        shown = panel._display_candidates(table)
        assert list(zip(shown["Material"], shown["Role"], shown["Origin"], strict=True)) == expected
    assert set(panel._display_candidates(sources.campaign_failure_rates(rows))["Reason"]) == {"Timed out"}
    view = panel._outcome_factory(sources.CampaignOutcomeCatalog(rows, designs={("transient_drying", "c"): plan}))()
    try:
        (origin,) = (widget for widget in _widgets(view.rows) if isinstance(widget, widgets.Dropdown) and widget.description == "Origin")
        origin.value = "replacement"  # Lentil and Kidney bean keep test designs only
        rates, reasons = [text for text in _slot_texts(view) if "<table" in text][1:]
        assert all(table.index(">Lentil<") < table.index(">Kidney bean<") for table in (rates, reasons))
    finally:
        plt.close("all")


def test_candidate_preview_heads_sampled_parameters_with_their_table_row_labels() -> None:
    """A sampled parameter or boundary column reads as its category and row label; an unlisted name stays visible without its underscores."""
    rows = (_candidate("transient_drying", "c", None, parameters={"T_flow_ref": 300.0, "bed.perturbations.amplitude": 0.1, "future_name": 1.0}),)
    shown = panel._display_candidates(sources.campaign_outcome_table(rows))
    assert {"Reference state · Flow temperature", "Bed perturb. · Amplitude", "future name"} <= set(shown.columns)
    assert not {"T_flow_ref", "bed.perturbations.amplitude", "future_name"} & set(shown.columns)


def test_failure_codes_read_as_display_text_and_stay_exact_in_the_evidence_details() -> None:
    """The candidate and failure-reason tables spell stage, class and reason as the EDA figures do; the evidence keeps the codes."""
    failure = {"stage": "validation", "class": "validation", "reason_code": "negative_water_state", "detail": "negative water content"}
    rows = (
        _candidate("transient_drying", "c", None, outcome="failed", failure=failure),
        _candidate("transient_drying", "c", None, case_id="case_0002"),
    )
    view = panel._outcome_factory(sources.CampaignOutcomeCatalog(rows))()
    try:
        texts = _slot_texts(view)
        tables = [text for text in texts if "<table" in text]
        assert len(tables) == 3  # candidates, rates and failure reasons
        assert all("Negative water state" in text for text in (tables[0], tables[2]))  # as token_display_label spells it in the figures
        assert not any("negative_water_state" in text for text in tables)
        assert any(text.startswith("<details>") and "negative_water_state" in text for text in texts)
    finally:
        plt.close("all")


def test_case_context_writes_input_directories_from_the_root_the_catalog_was_discovered_from(
    dataset_catalog: sources.GenerationInputDatasetCatalog,
) -> None:
    """The case-context view writes each input directory from the catalog's discovery root, not the environment's default root."""
    _key, state = _dataset_a_state(dataset_catalog)
    view = panel._view_factories(dataset_catalog, state)["case_context"]()
    try:
        (table,) = (text for text in _slot_texts(view) if "<table" in text)
        assert "../" not in table  # the tests' default storage root is another directory
        root = dataset_catalog.storage_root
        assert root is not None
        for case_key in (state.selection.case_a_key, state.selection.case_b_key):
            assert f">{dataset_catalog.load(case_key).case.directory.relative_to(root).as_posix()}<" in table
    finally:
        plt.close("all")


def test_case_context_names_the_prepared_inputs_and_dataset_tables_list_cases_as_rows(
    dataset_catalog: sources.GenerationInputDatasetCatalog,
) -> None:
    """Display rows and values, no execution status, a relative input directory; one row per actual case in the dataset tables."""
    key, _state = _dataset_a_state(dataset_catalog)
    dataset = dataset_catalog.dataset_diagnostics(key)
    context = generation_inputs.diagnostics.case_context_table(
        dataset.records[0], dataset, dataset.records[-1], dataset, storage_root=dataset_catalog.storage_root
    )
    assert all(label[0].isupper() for label in context.index)
    assert not any("executed" in str(value) for value in context.to_numpy().ravel())
    assert tuple(context.loc["Dataset"]) == (generation_inputs.diagnostics.dataset_caption(dataset),) * 2
    assert not any(Path(str(value)).is_absolute() for value in context.loc["Input directory (from storage root)"])
    cases = tuple(f"Case {record.case.case_index}" for record in dataset.records)
    for table in (generation_inputs.diagnostics.dataset_parameter_table(dataset), generation_inputs.diagnostics.dataset_field_summary_table(dataset)):
        assert all(tuple(frame.index) == cases for _section, frame in overview._cases_as_rows(table))


def test_transient_only_views_are_omitted_without_transient_inputs_and_kept_in_a_mixed_catalog() -> None:
    """A steady-only catalog offers no transient-only entry but keeps its boundary view; a mixed and a transient-only one keep them."""

    def keys(*profile_ids: str) -> set[str]:
        return {plot.key for section in presentation.sections_for_profiles(profile_ids) for plot in section.plots}

    transient_only = {"transient_schedules", "moisture_sorption"}
    steady = keys("steady_flow")
    assert not transient_only & steady
    assert "boundary_conditions" in steady  # the boundary view, left alone in its section, joins the case comparison
    assert transient_only <= keys("steady_flow", "transient_drying")
    assert transient_only | {"boundary_conditions"} <= keys("transient_drying")


def test_a_transient_campaign_without_a_dataset_keeps_the_transient_views_and_says_none_can_be_selected(
    dataset_catalog: sources.GenerationInputDatasetCatalog,
) -> None:
    """Steady Datasets beside a pending transient campaign keep both transient-only views, which say no transient Dataset exists yet."""
    steady = tuple(source for dataset in dataset_catalog.datasets if dataset.profile_id == "steady_flow" for source in dataset.publications)
    outcomes = sources.CampaignOutcomeCatalog((_candidate("transient_drying", "pending", None, outcome="pending"),))
    catalog = sources.GenerationInputDatasetCatalog(generation.cases.admission.InputSourceDiscovery(steady, ()), outcomes=outcomes)
    shell = panel.build_generation_input_eda_panel(datasets=catalog)
    try:
        next(widget for widget in _widgets((shell,)) if isinstance(widget, widgets.Button)).click()  # open the panel
        tabs = next(widget for widget in _widgets((shell,)) if isinstance(widget, widgets.Tab))
        assert {"transient_schedules", "moisture_sorption"} <= {key for section in tabs.children for *_entry, key in section.entries}
        view = panel._view_factories(catalog, generation_inputs.selection.GenerationInputSelectionState(catalog))["transient_schedules"]()
        assert any("no transient Dataset can be selected" in text for text in _slot_texts(view))
    finally:
        plt.close("all")


def test_comparison_figure_keys_its_case_identities_once_right_of_the_lower_panels(profile_records: ProfileRecords) -> None:
    """Captions name material, role and case; two materials take their colours; the key stands right of the panels at their top."""
    first, second = profile_records["steady_flow"]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    other = replace(mean, material_family="chickpea", batch_identity="b" * 64)
    assert layout.case_colors(mean, mean) == (layout.DATASET_A_COLOR, layout.DATASET_B_COLOR)
    figure = generation_inputs.plots.spatial.basic_comparison(first, mean, second, other, lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        figure.canvas.draw()
        captions = (layout.case_caption(first, mean, "A"), layout.case_caption(second, other, "B"))
        assert captions[1].startswith("Chickpea · Test · Case ")
        assert tuple(axis.get_title() for axis in figure.axes[:2]) == captions
        panels = figure.axes[6:8]  # three maps and their three bar slots lead; the distribution and the pressure profile follow
        assert panels[1].get_ylabel() == layout.quantity_axis_label("p_in_bc")
        assert all(axis.get_legend() is None for axis in panels)
        (key,) = figure.legends
        assert [text.get_text() for text in key.get_texts()] == [*captions, f"Mean A, n = {mean.case_count}", f"Mean B, n = {other.case_count}"]
        handle = key.legend_handles[1]
        assert isinstance(handle, Patch)
        assert handle.get_edgecolor()[:3] == pytest.approx(to_rgba(visual_semantics.material_color("chickpea"))[:3])
        assert not handle.get_fill()  # unfilled outlines: coincident cases never blend into a colour the key lacks
        assert [_dashed(entry) for entry in key.legend_handles[:2]] == [False, True]  # the key draws Case B dashed, as its bars
        extent = key.get_window_extent()
        assert extent.x0 >= max(axis.get_window_extent().x1 for axis in panels)
        assert extent.y1 == pytest.approx(max(axis.get_window_extent().y1 for axis in panels), abs=1.0)
    finally:
        plt.close(figure)


def test_uniform_fields_still_name_both_cases_in_the_shared_key(profile_records: ProfileRecords) -> None:
    """
    A uniform field draws each case as a line instead of bars; the shared key names both lines by their case captions.

    Two cases of one uniform value draw their lines at one place: Case B's is dashed, so Case A's still shows, and so is its key entry.
    """
    first, second = profile_records["steady_flow"]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    uniform = tuple(
        replace(record, fields=MappingProxyType({**record.fields, "eps_bed": np.full_like(record.fields["eps_bed"], 0.40)}))
        for record in (first, second)
    )
    figure = generation_inputs.plots.spatial.comparison_block(uniform[0], mean, uniform[1], mean, ("eps_bed",), title="Uniform", lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        (key,) = figure.legends
        captions = [layout.case_caption(record, mean, role) for record, role in zip(uniform, "AB", strict=True)]
        assert [text.get_text() for text in key.get_texts()][:2] == captions
        (panel,) = _distribution_axes(figure)
        assert [(line.get_label(), line.get_linestyle()) for line in panel.lines[:2]] == [(captions[0], "-"), (captions[1], "--")]
        assert [_dashed(handle) for handle in key.legend_handles[:2]] == [False, True]
    finally:
        plt.close(figure)


def test_a_field_writes_its_distribution_and_its_summary_rows_in_one_power_of_ten(profile_records: ProfileRecords) -> None:
    """The permeability rows of the summary take the factor of their distribution's axis, and their values are written in it."""
    first, second = profile_records["steady_flow"]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    figure = generation_inputs.plots.permeability.tensor_comparison(first, mean, second, mean, lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        histograms = _distribution_axes(figure)
        cells = next(axis for axis in figure.axes if axis.tables).tables[0].get_celld()
        statistics = generation_inputs.diagnostics.field_statistics(first)
        for row, (axis, quantity) in enumerate(zip(histograms, generation_inputs.plots.permeability.TENSOR_FIELDS, strict=True), start=1):
            power = _POWER.search(axis.get_xlabel())
            assert power is not None
            assert power.group(0) in cells[row, 0].get_text().get_text()
            exponent = int(power.group(0)[2:].translate(str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")))
            median = cast("float", statistics.loc[quantity, "median"])
            expected = float(f"{median / 10.0**exponent:.2e}")  # three significant digits in the row's power, rounded independently
            assert float(cells[row, 1].get_text().get_text().replace("\N{MINUS SIGN}", "-")) == expected
    finally:
        plt.close(figure)


def test_summary_cells_have_three_significant_digits_and_humidity_reads_in_the_figure_unit_style(profile_records: ProfileRecords) -> None:
    """Every number of the figure table keeps three significant digits, zeros included; its humidity unit is the one the axes write."""
    first, second = profile_records["transient_drying"]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    figure = generation_inputs.plots.moisture.moisture_comparison(first, mean, second, mean, lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        cells = next(axis for axis in figure.axes if axis.tables).tables[0].get_celld()
        numbers = [cell.get_text().get_text() for (row, column), cell in cells.items() if row and column]
        assert numbers
        assert all(text == "0" or len(text.lstrip("\N{MINUS SIGN}-").replace(".", "").lstrip("0")) == 3 for text in numbers), numbers
        labels = [cell.get_text().get_text() for (row, column), cell in cells.items() if row and not column]
        unit = f"[{field_labels.display_unit('1', field='phi_eq')}]"
        assert any(unit in label for label in labels)
        assert not any("[% RH]" in label for label in labels) or unit == "[% RH]"
    finally:
        plt.close(figure)


def test_a_scientific_row_rounds_notation_power_and_digits_alike_and_dashes_a_non_finite_value() -> None:
    """0.0009995 beside 1e-7 reads 1.00e-3, never 10.0e-4; a NaN beside them reads as a dash instead of failing; text stays text."""
    row = generation_inputs.plots.spatial._formatted_table_row((1e-7, 0.0009995, float("nan"), "n/a"), 0)
    assert [text.replace("\N{MINUS SIGN}", "-") for text in row] == ["1.00e-7", "1.00e-3", "\N{EN DASH}", "n/a"]


def test_a_summary_row_writes_all_its_numbers_in_one_notation(profile_records: ProfileRecords) -> None:
    """A shear row whose numbers lie on both sides of 10⁻³ in its power of ten writes each in its own power, three digits each."""
    first, second = profile_records["steady_flow"]
    records = []
    for record, median, peak in ((first, 4.5e-13, 3.0e-8), (second, 2.0e-11, -1.0e-8)):  # in the row's 10⁻⁸: 4.50e-5 and 0.00200
        values = np.full(record.fields["Kxy"].shape, median)
        values.flat[0] = peak  # the cell that sets the row's power of ten
        records.append(replace(record, fields=MappingProxyType({**record.fields, "Kxy": values})))
    mean = generation_inputs.diagnostics.build_dataset_diagnostics(tuple(records))
    figure = generation_inputs.plots.spatial.comparison_block(records[0], mean, records[1], mean, ("Kxy",), title="Kxy", lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        cells = next(axis for axis in figure.axes if axis.tables).tables[0].get_celld()
        written = [cells[1, column].get_text().get_text().replace("\N{MINUS SIGN}", "-") for column in range(1, 5)]
        assert all(re.fullmatch(r"-?[1-9]\.\d\de-?\d+", text) for text in written), written
        assert [float(text) for text in written] == pytest.approx([4.5e-5, 1.02e-3, 2.0e-3, 1.02e-3])
    finally:
        plt.close(figure)


def test_distributions_are_unfilled_outlines_and_label_the_count_axis_once_at_the_left(profile_records: ProfileRecords) -> None:
    """
    Coincident A/B bars never blend into an unkeyed colour, and only the leftmost distribution carries the count label.

    Case A's outlines are solid and Case B's dashed, so equal bin counts never hide Case A behind Case B.
    """
    first, second = profile_records["steady_flow"]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, second))
    figure = generation_inputs.plots.permeability.tensor_comparison(first, mean, first, mean, lock_scale=False)  # one case twice: equal bins
    assert isinstance(figure, Figure)
    try:
        panels = _distribution_axes(figure)
        assert [axis.get_ylabel() for axis in panels] == ["Counts", *[""] * (len(generation_inputs.plots.permeability.TENSOR_FIELDS) - 1)]
        assert all(axis.patches and not any(patch.get_fill() for patch in axis.patches) for axis in panels)
        styles = {(to_hex(patch.get_edgecolor()), _dashed(patch)) for axis in panels for patch in axis.patches}
        assert styles == {(to_hex(layout.DATASET_A_COLOR), False), (to_hex(layout.DATASET_B_COLOR), True)}
    finally:
        plt.close(figure)


def test_a_map_row_writes_its_three_bars_in_the_power_of_ten_of_its_distribution(profile_records: ProfileRecords) -> None:
    """Case B's permeability a decade below case A's: the A, B and B-minus-A bars of the row read in its distribution's power of ten."""
    first, second = profile_records["steady_flow"]
    lower = replace(second, fields=MappingProxyType({**second.fields, "Kxx": second.fields["Kxx"] * 0.1}))
    mean = generation_inputs.diagnostics.build_dataset_diagnostics((first, lower))
    figure = generation_inputs.plots.spatial.comparison_block(first, mean, lower, mean, ("Kxx",), title="Kxx", lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        bars = [axis.get_ylabel() for axis in figure.axes if axis.get_ylabel().startswith("[")]
        (histogram,) = _distribution_axes(figure)
        row_power = _POWER.search(histogram.get_xlabel())
        assert len(bars) == 3
        assert row_power is not None
        assert [match.group(0) if (match := _POWER.search(label)) else None for label in bars] == [row_power.group(0)] * 3
    finally:
        plt.close(figure)


def _distribution_axes(figure: Figure) -> list[Axes]:
    """Return the distribution panels of a comparison figure, left to right."""
    return [axis for axis in figure.axes if axis.get_title().endswith("distribution")]


def _dashed(artist: object) -> bool:
    """Return whether ``artist`` is drawn dashed, read from its dash pattern: a key's patch copies the dashes, not the style name."""
    return cast("Any", artist)._unscaled_dash_pattern[1] is not None


def _row_exponent(label: str) -> int:
    """Return the power of ten a unit bracket names, 0 without one."""
    power = _POWER.search(label)
    return int(power.group(0)[2:].translate(str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789"))) if power else 0


def test_constant_and_linear_rows_write_every_bar_label_in_the_row_power_and_only_an_exception_is_noted(profile_records: ProfileRecords) -> None:
    """
    A uniform permeability and a linear shear: each bar's unit and drawn labels are in the power of ten of its row's distribution.

    A permeability over seven decades keeps its decade labels, and its difference, finer than three decimals in the row's power,
    keeps its own; only these maps carry a note, which names the exception.
    """
    first, second = profile_records["steady_flow"]
    shape = first.fields["Kxx"].shape
    ramp = np.linspace(-1.0, 1.0, first.fields["Kxx"].size).reshape(shape)
    wide = np.geomspace(1.0e-12, 1.0e-5, ramp.size).reshape(shape)
    changes = (
        {"Kxx": np.full(shape, 2.0e-9), "Kxy": 2.0e-10 * ramp, "Kyy": wide},
        {"Kxx": np.full(shape, 3.0e-9), "Kxy": 1.0e-10 * ramp, "Kyy": wide * (1.0 + 1.0e-9 * ramp)},
    )
    records = [replace(record, fields=MappingProxyType({**record.fields, **change})) for record, change in zip((first, second), changes, strict=True)]
    mean = generation_inputs.diagnostics.build_dataset_diagnostics(tuple(records))
    quantities = ("Kxx", "Kxy", "Kyy")
    figure = generation_inputs.plots.spatial.comparison_block(records[0], mean, records[1], mean, quantities, title="Rows", lock_scale=False)
    assert isinstance(figure, Figure)
    try:
        figure.canvas.draw()
        maps = np.asarray(figure.axes[:9], dtype=object).reshape(3, 3)
        bars = np.asarray([axis for axis in figure.axes if axis.get_ylabel().startswith("[")], dtype=object).reshape(3, 3)
        rows = [_row_exponent(axis.get_xlabel()) for axis in _distribution_axes(figure)]
        assert rows == [-9, -10, -5]
        for row in range(2):
            for bar in bars[row]:
                assert _row_exponent(bar.get_ylabel()) == rows[row]
                for tick, label in zip(bar.get_yticks(), bar.get_yticklabels(), strict=True):
                    if text := label.get_text().replace("\N{MINUS SIGN}", "-"):
                        written = float(text) * 10.0 ** rows[row]
                        assert written == pytest.approx(tick, abs=0.51 * 10.0 ** (rows[row] - len(text.partition(".")[2])))
        assert [[label.get_text() for label in bar.get_yticklabels()] for bar in bars[0]] == [["2"], ["3"], ["1"]]
        notes = [[text.get_text() for text in axis.texts if text.get_gid() == "map-note"] for axis in maps.flat]
        assert notes[:6] == [[]] * 6
        assert notes[6:8] == [["Bar at decades (row in 10⁻⁵)"]] * 2
        assert _row_exponent(bars[2, 2].get_ylabel()) < -5
        assert notes[8] == [f"Bar in {_POWER.findall(bars[2, 2].get_ylabel())[0]} (row in 10⁻⁵)"]
    finally:
        plt.close(figure)


def test_one_signed_differences_and_crowded_maps_name_their_observed_range() -> None:
    """A difference of one sign, an anisotropy crowding a few bands and a near-uniform direction get a note; spread maps none."""
    positive = np.linspace(0.01, 0.04, 20)
    note = layout.one_sign_note(positive, layout.signed_norm(positive), "[-]")
    assert "≥ 0" in note
    assert "0.0100" in note
    assert "0.0400" in note
    both = np.linspace(-0.01, 0.04, 20)
    assert layout.one_sign_note(both, layout.signed_norm(both), "[-]") == ""
    crowded = np.concatenate((np.full(95, 1.02), np.linspace(1.0, 1.9, 5)))
    assert "1.02" in layout.compressed_note(crowded, layout.physical_norm("K_anisotropy", crowded))
    spread = np.linspace(1.0, 1.9, 100)
    assert layout.compressed_note(spread, layout.physical_norm("K_anisotropy", spread)) == ""
    assert "90°" in layout.axial_note(np.array([89.0, -89.0, 88.0, -88.5, 90.0] * 20))  # -89° and 89° are one direction
    assert layout.axial_note(np.linspace(-90.0, 90.0, 100)) == ""
