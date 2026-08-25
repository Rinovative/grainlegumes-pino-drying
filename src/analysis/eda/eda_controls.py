"""
eda_controls.py

Bind generated-output EDA widgets to shared catalog selection state.

Responsibilities:
  - Present one panel-level dataset selector over the complete catalog
  - Bind optional case navigation for child views without selector duplication
  - Expose only profile-compatible selected views and lazy frames to each view
  - Report concise compatibility and case-availability status

Design principles:
  - Widget values mirror canonical selection keys owned by eda_selection
  - Dataset changes rerender automatically without apply controls
  - Full batch and profile identities remain available as selector tooltips

This module does NOT:
  - Discover storage, choose plot channels, or render figures
  - Own task selection, physical-time alignment, or PDF export
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ipywidgets as widgets

from src.analysis import ui

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import pandas as pd

    from . import eda_selection as selection


class GeneratedOutputDatasetControl:
    """Bind the sole panel-level dataset selector to global EDA selection state."""

    def __init__(self, *, selection_state: selection.GeneratedOutputSelectionState) -> None:
        """Build one concise globally unique dataset checkbox group."""
        self._state = selection_state
        self._catalog = selection_state.catalog
        self._updating = True
        labels = self._catalog.labels()
        self._keys_by_label = {label: key for key, label in labels.items()}
        selected = set(selection_state.selection.dataset_keys)
        tooltips = {
            label: "; ".join(
                (
                    f"Canonical batch: {view.batch.batch_id}",
                    f"storage: {view.batch.batch_storage_name}",
                    f"profile: {view.simulation_profile}",
                    f"campaign purpose: {view.batch.campaign_purpose or 'unspecified'}",
                    f"source role: {view.batch.material_role or 'unspecified'}",
                    f"evaluation regime: {view.batch.evaluation_regime or 'unspecified'}",
                )
            )
            for label, key in self._keys_by_label.items()
            for view in (self._catalog.view(key),)
        }
        self.group = cast(
            "ui.components.CheckboxGroup",
            ui.components.ui_checkbox_datasets(
                dataset_names=list(self._keys_by_label),
                default_on=[label for label, key in self._keys_by_label.items() if key in selected],
                tooltips=tooltips,
                natural_width=True,
            ),
        )
        for checkbox in self.group.boxes.values():
            checkbox.observe(self._dataset_changed, names="value")
        self.widget = widgets.VBox(
            (widgets.HTML("<b>Datasets</b>"), cast("widgets.Widget", self.group)),
            layout=widgets.Layout(width="100%"),
        )
        selection_state.observe(self._state_changed)
        self._updating = False

    def _dataset_changed(self, _change: dict[str, object]) -> None:
        """Install checked stable-order dataset keys in shared state."""
        if self._updating:
            return
        checked = tuple(self._keys_by_label[label] for label, checkbox in self.group.boxes.items() if checkbox.value)
        self._state.select_datasets(checked)

    def _state_changed(self, current: selection.GeneratedOutputSelection) -> None:
        """Reflect programmatic dataset changes without widget feedback."""
        selected = set(current.dataset_keys)
        self._updating = True
        try:
            for label, checkbox in self.group.boxes.items():
                checkbox.value = self._keys_by_label[label] in selected
        finally:
            self._updating = False


class GeneratedOutputControls:
    """Bind one child-view case surface to shared capability-aware EDA selection."""

    def __init__(
        self,
        catalog: selection.GeneratedOutputEDACatalog,
        *,
        selection_state: selection.GeneratedOutputSelectionState,
        required_capabilities: Iterable[selection.GeneratedOutputCapability] = (),
        include_case: bool,
        case_companions: Sequence[widgets.Widget] = (),
    ) -> None:
        """Construct optional case controls for one explicitly compatible view."""
        if selection_state.catalog is not catalog:
            message = "Generated-output controls and state must use one catalog."
            raise ValueError(message)
        self._catalog = catalog
        self._state = selection_state
        self._required_capabilities = frozenset(required_capabilities)
        self._include_case = include_case
        self._callback: Callable[[], None] | None = None
        self._updating = True
        self._last_valid_case: int | None = None
        initial_cases = self._state.shared_case_numbers(required_capabilities=self._required_capabilities)
        if not initial_cases:
            compatible = self._state.selected_views(required_capabilities=self._required_capabilities)
            initial_cases = compatible[0].case_numbers if compatible else (0,)
        self.case, self.previous_case, self.next_case = ui.components.ui_step_case_index(
            case_numbers=initial_cases,
        )
        self.case_row = ui.components.ui_compact_case_row(
            self.case,
            self.previous_case,
            self.next_case,
        )
        self.status = widgets.HTML(
            layout=widgets.Layout(display="none"),
        )
        navigation_children: tuple[widgets.Widget, ...] = (self.case_row, *tuple(case_companions)) if include_case else ()
        self.navigation_row = widgets.HBox(
            navigation_children,
            layout=widgets.Layout(align_items="center", flex_flow="row wrap"),
        )
        children: list[widgets.Widget] = []
        if navigation_children:
            children.append(self.navigation_row)
        children.append(self.status)
        self.widget = widgets.VBox(tuple(children), layout=widgets.Layout(width="100%"))
        self.case.observe(self._case_changed, names="value")
        self.previous_case.on_click(lambda _button: self._step_case(-1))
        self.next_case.on_click(lambda _button: self._step_case(1))
        self._state.observe(self._state_changed)
        self._sync(self._state.selection)
        self._updating = False

    @property
    def required_capabilities(self) -> frozenset[selection.GeneratedOutputCapability]:
        """Return the explicit scientific capability requirements for this child view."""
        return self._required_capabilities

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Bind the active-aware render callback."""
        self._callback = callback

    @property
    def current_selection(self) -> selection.GeneratedOutputSelection:
        """Return the shared canonical selection reflected by these controls."""
        return self._state.selection

    def selected_views(self) -> tuple[selection.GeneratedOutputEDAView, ...]:
        """Return selected descriptors compatible with this view's requirements."""
        return self._state.selected_views(required_capabilities=self._required_capabilities)

    def selected_frames(self) -> dict[str, pd.DataFrame]:
        """Lazily load only selected frames compatible with this view."""
        return self._catalog.frames(
            self.current_selection.dataset_keys,
            required_capabilities=self._required_capabilities,
        )

    @property
    def selected_case_number(self) -> int | None:
        """Return the global case or one exact compatible-view fallback."""
        shared = self._shared_case_numbers()
        current = self.current_selection.case_number
        if current in shared:
            return current
        return shared[0] if shared else None

    def selected_case_ids(self) -> Mapping[str, str]:
        """Return compatible dataset labels mapped to the exact selected case ID."""
        case_number = self.selected_case_number
        if case_number is None:
            return {}
        return {view.label: view.case_id(case_number) for view in self.selected_views() if case_number in view.case_numbers}

    def availability_message(self) -> str:
        """Return concise status for selections omitted by this view's requirements."""
        selected_count = len(self.current_selection.dataset_keys)
        compatible_count = len(self.selected_views())
        if selected_count == 0:
            return "Select at least one dataset."
        if compatible_count == 0:
            return "No selected datasets provide this diagnostic."
        omitted = selected_count - compatible_count
        if omitted:
            return f"{compatible_count} compatible dataset(s); {omitted} unavailable for this diagnostic."
        return ""

    def activate(self) -> None:
        """Synchronize and render this child view from the current global state."""
        self._sync(self._state.selection)
        if self._callback is not None:
            self._callback()

    def _set_status_message(self, message: str) -> None:
        """Show one compatibility message or collapse its empty placeholder."""
        self.status.value = "" if not message else f"<p>{message}</p>"
        self.status.layout.display = "none" if not message else "flex"

    def _shared_case_numbers(self) -> tuple[int, ...]:
        """Return exact shared cases for datasets compatible with this child view."""
        return self._state.shared_case_numbers(required_capabilities=self._required_capabilities)

    def _sync_case(self, current: selection.GeneratedOutputSelection) -> None:
        """Synchronize typed case state and exact compatible-position bounds."""
        if not self._include_case:
            return
        shared = self._shared_case_numbers()
        selected = current.case_number if current.case_number is not None and current.case_number in shared else (shared[0] if shared else None)
        if selected is None:
            self.previous_case.disabled = True
            self.next_case.disabled = True
            return
        self.case.value = selected
        self._last_valid_case = selected
        position = shared.index(selected)
        self.previous_case.disabled = position == 0
        self.next_case.disabled = position == len(shared) - 1

    def _sync(self, current: selection.GeneratedOutputSelection) -> None:
        """Reflect one state update without widget feedback loops."""
        previous = self._updating
        self._updating = True
        try:
            self._sync_case(current)
            self._set_status_message(self.availability_message())
        finally:
            self._updating = previous

    def _state_changed(self, current: selection.GeneratedOutputSelection) -> None:
        """Synchronize and rerender after any global dataset or case update."""
        self._sync(current)
        if self._callback is not None:
            self._callback()

    def _case_changed(self, _change: dict[str, object]) -> None:
        """Accept one globally shared case number or restore the last valid value."""
        if self._updating or not self._include_case:
            return
        requested = int(self.case.value)
        try:
            self._state.select_case(
                requested,
                required_capabilities=self._required_capabilities,
            )
        except ValueError:
            self._updating = True
            try:
                if self._last_valid_case is not None:
                    self.case.value = self._last_valid_case
            finally:
                self._updating = False
            self._set_status_message(f"Case {requested} is unavailable for the selected datasets. Enter a shared case number or use the arrows.")
        else:
            self._set_status_message("")

    def _step_case(self, delta: int) -> None:
        """Move by one position in the exact compatible case-number intersection."""
        if self._updating:
            return
        shared = self._shared_case_numbers()
        current = self.selected_case_number
        if not shared or current not in shared:
            return
        position = shared.index(current)
        selected = shared[max(0, min(len(shared) - 1, position + delta))]
        self._state.select_case(
            selected,
            required_capabilities=self._required_capabilities,
        )
