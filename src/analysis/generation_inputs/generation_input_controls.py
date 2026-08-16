"""
===============================================================================
generation_input_controls.py
===============================================================================
Build local canonical-dataset selectors for generation-input views.
Responsibilities:
  - Bind Dataset A, Case A, Dataset B, and Case B controls
  - Default both datasets together and select the first two unique cases
  - Restrict A/B choices to compatible simulation profiles
  - Provide a local optional A/B map-scale lock and dataset-overview selector
Design principles:
  - Visible labels are concise while values retain canonical immutable keys
  - Selector changes rerender automatically without an update button
  - Case arrays and empirical means load only when the active view renders
This module does NOT:
  - Refresh discovery, construct tabs, render figures, or discover storage paths
  - Expose input-generation sources or lifecycle categories as datasets
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import ipywidgets as widgets

from . import generation_input_selection as shared_selection

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from src.analysis.generation_inputs import generation_input_diagnostics as diagnostics
    from src.analysis.generation_inputs import generation_input_sources as sources

DATASET_SELECTOR_WIDTH_PX = 360
DATASET_LABEL_WIDTH_PX = 78
CASE_LABEL_WIDTH_PX = 54
CASE_VALUE_WIDTH_PX = 63
CASE_SELECTOR_WIDTH_PX = CASE_LABEL_WIDTH_PX + CASE_VALUE_WIDTH_PX
CASE_STEP_WIDTH_PX = 40
CONTROL_ROW_VERTICAL_GAP_PX = 4
CONTROL_ROW_HORIZONTAL_GAP_PX = 6

_CONTROL_LAYOUT = widgets.Layout(
    display="flex",
    flex_flow="row wrap",
    align_items="center",
    grid_gap=(f"{CONTROL_ROW_VERTICAL_GAP_PX}px {CONTROL_ROW_HORIZONTAL_GAP_PX}px"),
    width="100%",
)
_DATASET_LAYOUT = widgets.Layout(
    width=f"{DATASET_SELECTOR_WIDTH_PX}px",
    max_width="100%",
    flex=f"0 0 {DATASET_SELECTOR_WIDTH_PX}px",
)
_CASE_LAYOUT = widgets.Layout(
    width=f"{CASE_SELECTOR_WIDTH_PX}px",
    flex=f"0 0 {CASE_SELECTOR_WIDTH_PX}px",
)
_STEP_LAYOUT = widgets.Layout(width=f"{CASE_STEP_WIDTH_PX}px")
_DATASET_STYLE = {"description_width": f"{DATASET_LABEL_WIDTH_PX}px"}
_CASE_STYLE = {"description_width": f"{CASE_LABEL_WIDTH_PX}px"}
_DATASET_KEY_LENGTH = 2
_CASE_KEY_LENGTH = 3


@dataclass(frozen=True, slots=True)
class SelectedComparison:
    """Bind selected cases to complete empirical dataset diagnostics."""

    case_a: diagnostics.GenerationInputDiagnostics
    mean_a: diagnostics.DatasetDiagnostics
    case_b: diagnostics.GenerationInputDiagnostics
    mean_b: diagnostics.DatasetDiagnostics
    same_dataset: bool
    lock_scale: bool


def _dropdown(
    *,
    description: str,
    layout: widgets.Layout,
) -> widgets.Dropdown:
    """Create one compact local dropdown with explicit label width."""
    return widgets.Dropdown(
        description=description,
        style=_DATASET_STYLE,
        layout=layout,
    )


def _case_control(*, description: str) -> widgets.BoundedIntText:
    """Create one compact directly editable numeric case selector."""
    return widgets.BoundedIntText(
        value=0,
        min=0,
        max=0,
        description=description,
        continuous_update=False,
        style=_CASE_STYLE,
        layout=_CASE_LAYOUT,
    )


def _step_button(description: str, tooltip: str) -> widgets.Button:
    """Create one compact case-navigation button."""
    return widgets.Button(
        description=description,
        tooltip=tooltip,
        layout=_STEP_LAYOUT,
    )


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
        self.dataset_a = _dropdown(description="Dataset A:", layout=_DATASET_LAYOUT)
        self.case_a = _case_control(description="Case A:")
        self.previous_a = _step_button("\u2190", "Previous case A")
        self.following_a = _step_button("\u2192", "Next case A")
        self.dataset_b = _dropdown(description="Dataset B:", layout=_DATASET_LAYOUT)
        self.case_b = _case_control(description="Case B:")
        self.previous_b = _step_button("\u2190", "Previous case B")
        self.following_b = _step_button("\u2192", "Next case B")
        self._case_keys_a: dict[int, sources.CaseKey] = {}
        self._case_keys_b: dict[int, sources.CaseKey] = {}
        self._last_valid_case_a = 0
        self._last_valid_case_b = 0
        self.scale_lock = (
            widgets.Checkbox(
                value=False,
                description="Lock A/B color scale",
                indent=False,
                style={"description_width": "initial"},
            )
            if include_scale_lock
            else None
        )
        self._apply_shared_selection(selection_state.selection)
        self.dataset_a.observe(self._on_dataset_a_change, names="value")
        self.dataset_b.observe(self._on_dataset_b_change, names="value")
        self.case_a.observe(
            lambda _change: self._on_case_change(self.case_a),
            names="value",
        )
        self.case_b.observe(
            lambda _change: self._on_case_change(self.case_b),
            names="value",
        )
        if self.scale_lock is not None:
            self.scale_lock.observe(self._on_value_change, names="value")
        self.previous_a.on_click(lambda _button: self._step_case(self.case_a, -1))
        self.following_a.on_click(lambda _button: self._step_case(self.case_a, 1))
        self.previous_b.on_click(lambda _button: self._step_case(self.case_b, -1))
        self.following_b.on_click(lambda _button: self._step_case(self.case_b, 1))
        self._selection_state.observe(self._on_shared_selection)

    @property
    def widget(self) -> widgets.VBox:
        """Return two compact selector rows and an optional local scale lock."""
        first = widgets.HBox(
            (self.dataset_a, self.case_a, self.previous_a, self.following_a),
            layout=_CONTROL_LAYOUT,
        )
        second = widgets.HBox(
            (self.dataset_b, self.case_b, self.previous_b, self.following_b),
            layout=_CONTROL_LAYOUT,
        )
        children: list[widgets.Widget] = [first, second]
        if self.scale_lock is not None:
            children.append(widgets.HBox((self.scale_lock,), layout=_CONTROL_LAYOUT))
        return widgets.VBox(tuple(children))

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback for semantic changes."""
        self._callback = callback

    @staticmethod
    def _dataset_key(dropdown: widgets.Dropdown) -> sources.DatasetKey | None:
        """Return one valid selected dataset key or None."""
        value = dropdown.value
        if isinstance(value, tuple) and len(value) == _DATASET_KEY_LENGTH and all(isinstance(item, str) for item in value):
            return cast("sources.DatasetKey", value)
        return None

    def _case_mapping(
        self,
        control: widgets.BoundedIntText,
    ) -> dict[int, sources.CaseKey]:
        """Return the immutable-key mapping owned by one numeric control."""
        if control is self.case_a:
            return self._case_keys_a
        if control is self.case_b:
            return self._case_keys_b
        message = "Case controls must belong to this A/B selector."
        raise ValueError(message)

    def _case_key(
        self,
        control: widgets.BoundedIntText,
    ) -> sources.CaseKey | None:
        """Return the canonical key behind one valid visible case number."""
        return self._case_mapping(control).get(int(control.value))

    def _last_valid_case(self, control: widgets.BoundedIntText) -> int:
        """Return the last valid number owned by one case control."""
        if control is self.case_a:
            return self._last_valid_case_a
        if control is self.case_b:
            return self._last_valid_case_b
        message = "Case controls must belong to this A/B selector."
        raise ValueError(message)

    def _set_last_valid_case(
        self,
        control: widgets.BoundedIntText,
        value: int,
    ) -> None:
        """Record the last valid typed number for one case control."""
        if control is self.case_a:
            self._last_valid_case_a = value
        elif control is self.case_b:
            self._last_valid_case_b = value
        else:
            message = "Case controls must belong to this A/B selector."
            raise ValueError(message)

    def _sync_case_buttons(self, control: widgets.BoundedIntText) -> None:
        """Disable navigation arrows at the current available-case bounds."""
        numbers = tuple(self._case_mapping(control))
        if control is self.case_a:
            previous, following = self.previous_a, self.following_a
        else:
            previous, following = self.previous_b, self.following_b
        if not numbers or int(control.value) not in numbers:
            previous.disabled = True
            following.disabled = True
            return
        position = numbers.index(int(control.value))
        previous.disabled = position == 0
        following.disabled = position == len(numbers) - 1

    def _step_case(self, control: widgets.BoundedIntText, delta: int) -> None:
        """Move one numeric selector through admitted sparse case numbers."""
        numbers = tuple(self._case_mapping(control))
        if not numbers or int(control.value) not in numbers:
            return
        position = numbers.index(int(control.value))
        target = max(0, min(len(numbers) - 1, position + delta))
        control.value = numbers[target]

    def _sync_case_control(
        self,
        control: widgets.BoundedIntText,
        dataset_key: sources.DatasetKey,
        selected_key: sources.CaseKey,
    ) -> None:
        """Rebuild one numeric widget from canonical admitted case keys."""
        mapping = self._case_mapping(control)
        mapping.clear()
        for label, case_key in self._catalog.case_options(dataset_key):
            try:
                number = int(label)
            except ValueError as error:
                message = f"Generation-input case label must be numeric, got {label!r}."
                raise ValueError(message) from error
            if number in mapping:
                message = f"Generation-input dataset contains duplicate case number {number}."
                raise ValueError(message)
            mapping[number] = case_key
        selected_numbers = tuple(number for number, case_key in mapping.items() if case_key == selected_key)
        if len(selected_numbers) != 1:
            message = "Shared generation-input case is absent from its selected dataset."
            raise ValueError(message)
        numbers = tuple(mapping)
        selected = selected_numbers[0]
        control.disabled = not numbers
        minimum = min(numbers, default=0)
        maximum = max(numbers, default=0)
        if minimum > control.max:
            control.max = maximum
            control.min = minimum
        else:
            control.min = minimum
            control.max = maximum
        control.value = selected
        self._set_last_valid_case(control, selected)
        self._sync_case_buttons(control)

    def _apply_shared_selection(
        self,
        value: shared_selection.GenerationInputSelection,
    ) -> None:
        """Synchronize every local widget under one explicit observer guard."""
        self._updating = True
        try:
            self.dataset_a.options = self._catalog.dataset_options()
            self.dataset_a.value = value.dataset_a_key
            self.dataset_b.options = self._catalog.dataset_options(profile_ids=(value.dataset_a_key[0],))
            self.dataset_b.value = value.dataset_b_key
            self._sync_case_control(
                self.case_a,
                value.dataset_a_key,
                value.case_a_key,
            )
            self._sync_case_control(
                self.case_b,
                value.dataset_b_key,
                value.case_b_key,
            )
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

    def _on_dataset_a_change(self, _change: object) -> None:
        """Publish one local Dataset A change to shared state."""
        if self._updating:
            return
        dataset_key = self._dataset_key(self.dataset_a)
        if dataset_key is not None:
            self._selection_state.select_dataset_a(dataset_key)

    def _on_dataset_b_change(self, _change: object) -> None:
        """Publish one local Dataset B change to shared state."""
        if self._updating:
            return
        dataset_key = self._dataset_key(self.dataset_b)
        if dataset_key is not None:
            self._selection_state.select_dataset_b(dataset_key)

    def _on_value_change(self, _change: object) -> None:
        """Render after a genuinely view-local scale-lock change."""
        self._notify()

    def _on_case_change(self, control: widgets.BoundedIntText) -> None:
        """Publish valid typed cases and restore rejected sparse numbers."""
        if self._updating:
            return
        case_key = self._case_key(control)
        if case_key is None:
            self._updating = True
            try:
                control.value = self._last_valid_case(control)
            finally:
                self._updating = False
            self._sync_case_buttons(control)
            return
        if control is self.case_a:
            self._selection_state.select_case_a(case_key)
        else:
            self._selection_state.select_case_b(case_key)


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
        self.dataset = _dropdown(
            description="Dataset:",
            layout=_DATASET_LAYOUT,
        )
        options = catalog.dataset_options(profile_ids=profile_ids)
        self.dataset.options = options
        self.dataset.value = options[0][1] if options else None
        self.dataset.observe(self._on_value_change, names="value")

    @property
    def widget(self) -> widgets.HBox:
        """Return the compact local dataset selector."""
        return widgets.HBox(
            (self.dataset,),
            layout=_CONTROL_LAYOUT,
        )

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the automatic render callback for dataset changes."""
        self._callback = callback

    def selected_diagnostics(self) -> diagnostics.DatasetDiagnostics:
        """Load all unique cases and empirical summaries for the dataset."""
        value = self.dataset.value
        if not isinstance(value, tuple) or len(value) != _DATASET_KEY_LENGTH or not all(isinstance(item, str) for item in value):
            msg = "A valid generation-input dataset is required."
            raise ValueError(msg)
        return self._catalog.dataset_diagnostics(cast("sources.DatasetKey", value))

    def _on_value_change(self, _change: object) -> None:
        """Render after the selected dataset changes."""
        if self._callback is not None:
            self._callback()
