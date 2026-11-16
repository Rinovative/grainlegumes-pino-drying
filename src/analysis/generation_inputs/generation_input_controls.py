"""Build local canonical-dataset selectors for generation-input views.

Responsibilities:
  - Bind Dataset A, Case A, Dataset B, and Case B selectors and navigators to the shared selection
  - Default both datasets together and select the first two unique cases
  - Restrict Dataset B to the simulation profile of Dataset A
  - Provide a local optional A/B map-scale lock, a dataset-overview selector, atlas field checkboxes, campaign-outcome filters, and the
    Campaign A/B choice of the paired template comparison

Design principles:
  - Visible labels are concise while values retain canonical immutable keys
  - Selector changes rerender automatically without an update button
  - Case arrays and empirical means load only when the active view renders
  - Every control comes from the control standard and is placed in panel rows, datasets in collapsed groups by sampling regime

This module does NOT:
  - Refresh discovery, construct tabs, render figures, or discover storage paths
  - Expose input-generation sources or lifecycle categories as datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from src.analysis import presentation, ui

from . import generation_input_diagnostics as diagnostics
from . import generation_input_selection as shared_selection
from . import generation_input_sources as sources
from . import generation_input_table_schema as table_schema
from . import generation_input_template_comparison as template
from .plots import generation_input_plot_atlas as atlas_plots

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import ipywidgets as widgets

type ViewPopover = tuple[widgets.ToggleButton, widgets.DOMWidget]


@dataclass(frozen=True, slots=True)
class SelectedComparison:
    """Bind selected cases to complete empirical dataset diagnostics."""

    case_a: diagnostics.GenerationInputDiagnostics
    mean_a: diagnostics.DatasetDiagnostics
    case_b: diagnostics.GenerationInputDiagnostics
    mean_b: diagnostics.DatasetDiagnostics
    same_dataset: bool
    lock_scale: bool


def _dataset_block(
    catalog: sources.GenerationInputDatasetCatalog,
    noun: str,
    selected: sources.DatasetKey | None,
    *,
    profile_ids: Iterable[str] | None = None,
) -> ui.selectors.ChoiceBlock:
    """Return a one-of dataset block over the catalog's datasets, grouped by sampling regime."""
    labels = {key: label for label, key in catalog.dataset_options(profile_ids=profile_ids)}
    groups: dict[str, list[tuple[str, sources.DatasetKey]]] = {}
    for dataset in catalog.datasets:
        key = sources.dataset_key(dataset)
        if key in labels:
            groups.setdefault(dataset.sampling_regime, []).append((labels[key], key))
    return ui.selectors.ChoiceBlock(
        noun,
        [(presentation.display_labels.regime_display_label(regime), items) for regime, items in groups.items()],
        multiple=False,
        selected=[selected] if selected in labels else [],
    )


def _dataset_key(block: ui.selectors.ChoiceBlock) -> sources.DatasetKey:
    """Return the canonical key of a one-of dataset block's selection."""
    return cast("sources.DatasetKey", block.selected()[0])


def _case_numbers(catalog: sources.GenerationInputDatasetCatalog, dataset_key: sources.DatasetKey) -> dict[int, sources.CaseKey]:
    """Return the immutable case key behind every admitted numeric case label of one dataset."""
    numbers: dict[int, sources.CaseKey] = {}
    for label, case_key in catalog.case_options(dataset_key):
        try:
            number = int(label)
        except ValueError as error:
            message = f"Generation-input case label must be numeric, got {label!r}."
            raise ValueError(message) from error
        if number in numbers:
            message = f"Generation-input dataset contains duplicate case number {number}."
            raise ValueError(message)
        numbers[number] = case_key
    return numbers


class _CaseSelector:
    """Bind one case navigator to the admitted sparse case numbers of a dataset and publish the cases a user selects."""

    def __init__(
        self,
        catalog: sources.GenerationInputDatasetCatalog,
        dataset_key: sources.DatasetKey,
        case_key: sources.CaseKey,
        *,
        description: str,
        status: ui.controls.StatusLine,
        publish: Callable[[sources.CaseKey], None],
    ) -> None:
        """Create the navigator on ``case_key``; ``publish`` receives every case a user selects."""
        self._catalog = catalog
        self._publish = publish
        self._keys: dict[int, sources.CaseKey] = {}
        self._updating = False
        numbers, selected = self._resolve(dataset_key, case_key)
        self.navigator = ui.navigation.CaseNavigator(numbers, value=selected, description=description, status=status, on_change=self._changed)

    def sync(self, dataset_key: sources.DatasetKey, case_key: sources.CaseKey) -> None:
        """Show the case numbers of ``dataset_key`` with ``case_key`` selected, without publishing."""
        numbers, selected = self._resolve(dataset_key, case_key)
        self._updating = True
        try:
            # An unchanged navigator is left alone, so the warning of a corrected typed case stays on the status line.
            if self.navigator.case_ids != numbers:
                self.navigator.rebind(numbers, preserve=False, notify=False)
            if self.navigator.value != selected:
                self.navigator.step(numbers.index(selected) - self.navigator.position)
        finally:
            self._updating = False

    def _resolve(self, dataset_key: sources.DatasetKey, case_key: sources.CaseKey) -> tuple[tuple[int, ...], int]:
        """Rebuild the number-to-key mapping and return the numbers and the number of ``case_key``."""
        self._keys = _case_numbers(self._catalog, dataset_key)
        selected = [number for number, key in self._keys.items() if key == case_key]
        if len(selected) != 1:
            message = "Shared generation-input case is absent from its selected dataset."
            raise ValueError(message)
        return tuple(self._keys), selected[0]

    def _changed(self, number: int) -> None:
        """Publish a user-selected case number as its canonical case key."""
        if not self._updating:
            self._publish(self._keys[number])


class FieldAtlasControls:
    """Bind a single admitted case and independently retained spatial fields."""

    def __init__(
        self,
        catalog: sources.GenerationInputDatasetCatalog,
        selection_state: shared_selection.GenerationInputSelectionState,
    ) -> None:
        """Use the common Case A selection and load only that case, never the dataset means."""
        if selection_state.catalog is not catalog:
            message = "Atlas controls require the selection state's admitted catalog."
            raise ValueError(message)
        self._catalog = catalog
        self._state = selection_state
        self._callback: Callable[[], None] | None = None
        self._updating = False
        selection = selection_state.selection
        self.status = ui.controls.StatusLine()
        self.dataset = _dataset_block(catalog, "Datensätze", selection.dataset_a_key)
        self._case = _CaseSelector(
            catalog, selection.dataset_a_key, selection.case_a_key, description="Case", status=self.status, publish=selection_state.select_case_a
        )
        self._loaded_key = selection.case_a_key
        self._atlas = atlas_plots.prepare_field_atlas(catalog.load(self._loaded_key))
        self.fields = ui.controls.ChannelChecks(tuple(self._atlas.fields), labels=self._atlas.labels, title="Spatial fields")
        self.rows = (ui.panel.row(self.dataset.header, self._case.navigator.widget), ui.panel.row(self.fields.widget))
        self.popovers: tuple[ViewPopover, ...] = ((self.dataset.header, self.dataset.body),)
        self.dataset.observe(self._dataset_changed)
        self.fields.observe(self._changed)
        selection_state.observe(self._synchronize)

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind automatic rendering without eagerly opening scientific arrays."""
        self._callback = callback

    def _changed(self) -> None:
        """Notify the active view after a complete semantic state update."""
        if self._callback is not None and not self._updating:
            self._callback()

    def _dataset_changed(self) -> None:
        """Delegate canonical membership reconciliation to the shared owner."""
        if not self._updating:
            self._state.select_dataset_a(_dataset_key(self.dataset))

    def _synchronize(self, selection: shared_selection.GenerationInputSelection) -> None:
        """Update controls without recursive events or loading unrelated cases."""
        self._updating = True
        try:
            self.dataset.select([selection.dataset_a_key])
            self._case.sync(selection.dataset_a_key, selection.case_a_key)
        finally:
            self._updating = False
        self._changed()

    def selected_atlas(self) -> atlas_plots.GeneratedFieldAtlas:
        """Prepare arrays and full-grid scales only when the selected case changes."""
        key = self._state.selection.case_a_key
        if self._loaded_key != key:
            self._atlas = atlas_plots.prepare_field_atlas(self._catalog.load(key))
            self._loaded_key = key
            removed = self.fields.rebind(tuple(self._atlas.fields), labels=self._atlas.labels)
            if removed:
                self.status.set(f"Unavailable fields removed: {', '.join(removed)}.", "warning")
        return self._atlas


class PairCaseControls:
    """Bind view-local controls to one shared canonical A/B selection."""

    def __init__(
        self,
        catalog: sources.GenerationInputDatasetCatalog,
        *,
        selection_state: shared_selection.GenerationInputSelectionState,
        include_scale_lock: bool = False,
    ) -> None:
        """Initialize synchronized local controls from authoritative session state."""
        if not isinstance(
            selection_state,
            shared_selection.GenerationInputSelectionState,
        ):
            message = "PairCaseControls requires a GenerationInputSelectionState."
            raise TypeError(message)
        if selection_state.catalog is not catalog:
            message = "PairCaseControls state must use the supplied dataset catalog."
            raise ValueError(message)
        self._catalog = catalog
        self._selection_state = selection_state
        self._callback: Callable[[], None] | None = None
        self._updating = True
        selection = selection_state.selection
        self.status = ui.controls.StatusLine()
        self.dataset_a = _dataset_block(catalog, "Datensatz A", selection.dataset_a_key)
        self._cases_a = _CaseSelector(
            catalog, selection.dataset_a_key, selection.case_a_key, description="Case A", status=self.status, publish=selection_state.select_case_a
        )
        self.case_a = self._cases_a.navigator
        # Dataset B offers only Dataset A's profile, so every profile has its own block and the matching one is shown.
        self._datasets_b = {
            profile: _dataset_block(catalog, "Datensatz B", selection.dataset_b_key, profile_ids=(profile,)) for profile in catalog.profiles
        }
        self._cases_b = _CaseSelector(
            catalog, selection.dataset_b_key, selection.case_b_key, description="Case B", status=self.status, publish=selection_state.select_case_b
        )
        self.case_b = self._cases_b.navigator
        self.scale_lock = ui.controls.scale_lock_checkbox(value=False, description="Lock A/B color scale") if include_scale_lock else None
        rows = [
            ui.panel.row(self.dataset_a.header, self.case_a.widget),
            ui.panel.row(*(block.header for block in self._datasets_b.values()), self.case_b.widget),
        ]
        if self.scale_lock is not None:
            rows.append(ui.panel.row(self.scale_lock))
        self.rows = tuple(rows)
        self.popovers: tuple[ViewPopover, ...] = tuple((block.header, block.body) for block in (self.dataset_a, *self._datasets_b.values()))
        self._apply_shared_selection(selection)
        self.dataset_a.observe(self._on_dataset_a_change)
        for block in self._datasets_b.values():
            block.observe(self._on_dataset_b_change)
        if self.scale_lock is not None:
            self.scale_lock.observe(lambda _change: self._notify(), names="value")
        self._selection_state.observe(self._on_shared_selection)

    @property
    def dataset_b(self) -> ui.selectors.ChoiceBlock:
        """Return the Dataset B block of Dataset A's simulation profile."""
        return self._datasets_b[self._selection_state.selection.dataset_a_key[0]]

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback for semantic changes."""
        self._callback = callback

    def _apply_shared_selection(
        self,
        value: shared_selection.GenerationInputSelection,
    ) -> None:
        """Synchronize every local control under one explicit observer guard."""
        self._updating = True
        try:
            self.dataset_a.select([value.dataset_a_key])
            for profile, block in self._datasets_b.items():
                ui.controls.set_visible(block.header, profile == value.dataset_a_key[0])
            self._datasets_b[value.dataset_a_key[0]].select([value.dataset_b_key])
            self._cases_a.sync(value.dataset_a_key, value.case_a_key)
            self._cases_b.sync(value.dataset_b_key, value.case_b_key)
        finally:
            self._updating = False

    def _on_shared_selection(
        self,
        value: shared_selection.GenerationInputSelection,
    ) -> None:
        """Reflect authoritative changes and rerender this created view once."""
        self._apply_shared_selection(value)
        self._notify()

    def selected_comparison(self) -> SelectedComparison:
        """Load selected cases and all unique cases behind both means."""
        value = self._selection_state.selection
        return SelectedComparison(
            case_a=self._catalog.load(value.case_a_key),
            mean_a=self._catalog.dataset_diagnostics(value.dataset_a_key),
            case_b=self._catalog.load(value.case_b_key),
            mean_b=self._catalog.dataset_diagnostics(value.dataset_b_key),
            same_dataset=value.dataset_a_key == value.dataset_b_key,
            lock_scale=(bool(self.scale_lock.value) if self.scale_lock is not None else False),
        )

    def _notify(self) -> None:
        """Invoke the bound renderer outside synchronized widget updates."""
        if not self._updating and self._callback is not None:
            self._callback()

    def _on_dataset_a_change(self) -> None:
        """Publish one local Dataset A change to shared state."""
        if not self._updating:
            self._selection_state.select_dataset_a(_dataset_key(self.dataset_a))

    def _on_dataset_b_change(self) -> None:
        """Publish one local Dataset B change to shared state."""
        if not self._updating:
            self._selection_state.select_dataset_b(_dataset_key(self.dataset_b))


class DatasetControls:
    """Own one local dataset selector for dataset-overview views."""

    def __init__(
        self,
        catalog: sources.GenerationInputDatasetCatalog,
        *,
        profile_ids: Iterable[str] | None = None,
    ) -> None:
        """Initialize one dataset selector with an immediate valid default."""
        self._catalog = catalog
        self._callback: Callable[[], None] | None = None
        options = catalog.dataset_options(profile_ids=profile_ids)
        self.dataset = _dataset_block(catalog, "Datensätze", options[0][1] if options else None, profile_ids=profile_ids)
        self.status = ui.controls.StatusLine()
        self.rows = (ui.panel.row(self.dataset.header),)
        self.popovers: tuple[ViewPopover, ...] = ((self.dataset.header, self.dataset.body),)
        self.dataset.observe(self._on_value_change)

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback for dataset changes."""
        self._callback = callback

    def selected_diagnostics(self) -> diagnostics.DatasetDiagnostics:
        """Load all unique cases and empirical summaries for the dataset."""
        selected = self.dataset.selected()
        if len(selected) != 1:
            msg = "A valid generation-input dataset is required."
            raise ValueError(msg)
        return self._catalog.dataset_diagnostics(cast("sources.DatasetKey", selected[0]))

    def _on_value_change(self) -> None:
        """Render after the selected dataset changes."""
        if self._callback is not None:
            self._callback()


class CampaignOutcomeControls:
    """Filter a single discovered campaign checkpoint without storage reads."""

    def __init__(self, catalog: sources.CampaignOutcomeCatalog) -> None:
        """Retain exact candidate selection independently of successful Dataset pools."""
        self.catalog = catalog
        dropdown = ui.controls.dropdown
        campaigns = tuple(sorted({(row["task"], row["source_name"]) for row in catalog.rows}))
        self.campaign = dropdown([(f"{task}/{name}", (task, name)) for task, name in campaigns], description="Campaign")
        self.material = dropdown(
            [("All materials", None), *((name, name) for name in sorted({row["material_family"] for row in catalog.rows}))], description="Material"
        )
        self.outcome = dropdown(
            [("All outcomes", None), ("Successful", "successful"), ("Failed", "failed"), ("Pending", "pending")], description="Outcome"
        )
        self.origin = dropdown(
            [("All origins", None), *((name, name) for name in sorted({row["origin"] for row in catalog.rows}))], description="Origin"
        )
        parameters = {
            (name, str(unit))
            for row in catalog.rows
            for name, unit in (row.get("parameter_units") or {}).items()
            if isinstance((row.get("parameters") or {}).get(name), (int, float))
        }
        # No parameter is drawn until chosen; the grids follow the table sections of the parameters.
        self.parameters = ui.selectors.ChoiceBlock(
            "Parameter",
            [
                (section, [(f"{name} [{diagnostics.display_unit(name, unit)}]", (name, unit)) for name, unit in group])
                for section, group in table_schema.grouped_parameters(parameters).items()
            ],
            multiple=True,
            selected=[],
            per_line=True,  # the names are long
        )
        # Candidate evidence follows the filters, so its width covers every candidate the catalog can offer.
        self.case = dropdown(self._case_options(), description="Evidence", width_texts=[_candidate_label(row) for row in catalog.rows])
        self.status = ui.controls.StatusLine()
        self.rows = (
            ui.panel.row(self.campaign, self.material),
            ui.panel.row(self.outcome, self.origin),
            ui.panel.row(self.case, self.parameters.header),
        )
        self.popovers: tuple[ViewPopover, ...] = ((self.parameters.header, self.parameters.body),)
        self._callback: Callable[[], None] | None = None
        self._updating = False
        for widget in (self.campaign, self.material, self.outcome, self.origin):
            widget.observe(self._filter_changed, names="value")
        self.case.observe(self._value_changed, names="value")
        self.parameters.observe(lambda: self._value_changed({}))

    def chosen_parameters(self) -> tuple[table_schema.Parameter, ...]:
        """Return the chosen ``(name, unit)`` parameters in table order."""
        return cast("tuple[table_schema.Parameter, ...]", self.parameters.selected())

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind one active-view update."""
        self._callback = callback

    def candidates(self, *, all_outcomes: bool = False) -> tuple[dict[str, Any], ...]:
        """Return candidate metadata rows on exact user-selected campaign/material scope."""
        if self.campaign.value is None:
            return ()
        task, name = self.campaign.value
        return sources.query_campaign_outcomes(
            self.catalog,
            task=task,
            source_name=name,
            material_family=self.material.value,
            outcome=None if all_outcomes else self.outcome.value,
            origin=self.origin.value,
        )

    def _case_options(self) -> list[tuple[str, tuple[str, str]]]:
        """Return the evidence choices of the candidates that match the filters."""
        return [(_candidate_label(row), (row["material_family"], row["case_id"])) for row in self.candidates()]

    def _rebind_case(self) -> None:
        """Preserve an exact selected candidate when filtering retains it."""
        options = self._case_options()
        self._updating = True
        try:
            _rebind(self.case, options)
        finally:
            self._updating = False

    def _filter_changed(self, _change: dict[str, Any]) -> None:
        """Rebind visible candidate evidence, then update only the active view."""
        self._rebind_case()
        self._value_changed({})

    def _value_changed(self, _change: dict[str, Any]) -> None:
        """Refresh a selected evidence record or optional parameter plot."""
        if not self._updating and self._callback is not None:
            self._callback()


class TemplateComparisonControls:
    """Choose Campaign A and a Campaign B of its task that shares designs with it."""

    def __init__(self, catalog: sources.CampaignOutcomeCatalog, partners: Mapping[template.Campaign, tuple[template.Campaign, ...]]) -> None:
        """Offer the campaigns of ``partners``, the result of ``comparable_campaigns``; Campaign B starts on the first partner of A."""
        self.catalog = catalog
        self._partners = partners
        self._labels = {campaign: f"{campaign[0]}/{campaign[1]}" for campaign in self._partners}
        texts = list(self._labels.values())
        self.campaign_a = ui.controls.dropdown([(label, campaign) for campaign, label in self._labels.items()], description="Campaign A")
        self.campaign_b = ui.controls.dropdown(self._b_options(), description="Campaign B", width_texts=texts)
        self.status = ui.controls.StatusLine()
        self.rows = (ui.panel.row(self.campaign_a, self.campaign_b),)
        self.popovers: tuple[ViewPopover, ...] = ()
        self._callback: Callable[[], None] | None = None
        self._updating = False
        self.campaign_a.observe(self._a_changed, names="value")
        self.campaign_b.observe(self._notify, names="value")

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind one active-view update."""
        self._callback = callback

    def comparison(self) -> template.TemplateComparison:
        """Pair the selected campaigns over their quantities."""
        campaign_a, campaign_b = (cast("template.Campaign", widget.value) for widget in (self.campaign_a, self.campaign_b))
        return template.compare_templates(self.catalog.rows, campaign_a, campaign_b, quantities=template.QUANTITIES)

    def _b_options(self) -> list[tuple[str, template.Campaign]]:
        """Return the partners of the selected Campaign A."""
        return [(self._labels[campaign], campaign) for campaign in self._partners[cast("template.Campaign", self.campaign_a.value)]]

    def _a_changed(self, _change: dict[str, Any]) -> None:
        """Offer the partners of the new Campaign A, keep Campaign B when it is one of them, then update once."""
        options = self._b_options()
        self._updating = True
        try:
            _rebind(self.campaign_b, options)
        finally:
            self._updating = False
        self._notify({})

    def _notify(self, _change: dict[str, Any]) -> None:
        """Update the active view after a selection."""
        if not self._updating and self._callback is not None:
            self._callback()


def _rebind(dropdown: widgets.Dropdown, options: list[tuple[str, Any]]) -> None:
    """Replace the options of ``dropdown``; its value stays when it is still offered, else it moves to the first option."""
    previous = dropdown.value
    dropdown.options = options
    if previous in {value for _label, value in options}:
        dropdown.value = previous


def _candidate_label(row: dict[str, Any]) -> str:
    """Return the evidence choice text of one campaign candidate."""
    return f"{row['material_family']}/{row['case_id']} — {row['outcome']}"
