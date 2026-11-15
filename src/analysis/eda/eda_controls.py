"""Bind generated-output EDA widgets to shared catalog selection state.

Responsibilities:
  - Present one panel-level dataset selector over the complete catalog, grouped by sampling regime
  - Bind the case navigator of one child view without selector duplication
  - Expose only profile-compatible selected views and lazy frames to each view
  - Report concise compatibility and case-availability status

Design principles:
  - Widget values mirror canonical selection keys owned by eda_selection
  - Dataset changes rerender automatically without apply controls
  - A deactivated child view skips shared-state sync and rendering until activated
  - Full batch and profile identities remain available as selector tooltips

This module does NOT:
  - Discover storage, choose plot channels, or render figures
  - Own task selection, physical-time alignment, or PDF export
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from src.analysis import presentation, ui

from . import eda_selection as selection

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    import pandas as pd


def _inventory(catalog: selection.GeneratedOutputEDACatalog) -> presentation.dataset_selection.DatasetInventory:
    """Return the catalog's datasets grouped by sampling regime; a published package groups under its task."""
    items = []
    group_labels: dict[str, str] = {}
    for view in catalog.views:
        if isinstance(view, selection.GeneratedOutputEDAView):
            group, material = view.batch.sampling_regime, view.batch.material_family
            group_labels[group] = presentation.display_labels.regime_display_label(group)
        else:
            group, material = view.task_id, ""
            group_labels[group] = presentation.display_labels.task_display_label(group)
        items.append(presentation.dataset_selection.DatasetItem(view.key, material, group, view.label))
    return presentation.dataset_selection.DatasetInventory(items, group_order=tuple(group_labels), group_labels=group_labels, poolable=False)


def dataset_selector(
    selection_state: selection.GeneratedOutputSelectionState, *, commit: Callable[[tuple[str, ...]], None] | None = None
) -> ui.selectors.DatasetSelector:
    """
    Build the sole panel-level dataset selector and keep it and the shared selection in step in both directions.

    A user's choice goes to ``commit`` (default: the selection state), so the panel can finish its own work after the
    notification of the state has returned.
    """
    catalog = selection_state.catalog
    selector = ui.selectors.DatasetSelector(_inventory(catalog), multiple=True, selected=selection_state.selection.dataset_keys)
    for view in catalog.views:
        selector.boxes[view.key].tooltip = view.tooltip
    apply = selection_state.select_datasets if commit is None else commit
    selector.observe(lambda: apply(cast("tuple[str, ...]", selector.selected())))
    selection_state.observe(lambda current: selector.set_state(current.dataset_keys, pooled=False))
    return selector


class GeneratedOutputControls:
    """Bind one child-view case surface to shared capability-aware EDA selection."""

    def __init__(
        self,
        catalog: selection.GeneratedOutputEDACatalog,
        *,
        selection_state: selection.GeneratedOutputSelectionState,
        required_capabilities: Iterable[selection.GeneratedOutputCapability] = (),
        include_case: bool,
    ) -> None:
        """Construct the optional case navigator for one explicitly compatible view."""
        if selection_state.catalog is not catalog:
            message = "Generated-output controls and state must use one catalog."
            raise ValueError(message)
        self._catalog = catalog
        self._state = selection_state
        self._required_capabilities = frozenset(required_capabilities)
        self._include_case = include_case
        self._callback: Callable[[], None] | None = None
        self._active = True
        self._updating = True
        self._availability_shown = False
        self.status = ui.controls.StatusLine()
        self._case: ui.navigation.CaseNavigator | None = None
        if include_case:
            initial_cases = self._state.shared_case_numbers(required_capabilities=self._required_capabilities)
            if not initial_cases:
                compatible = self._state.selected_views(required_capabilities=self._required_capabilities)
                initial_cases = compatible[0].case_numbers if compatible else (0,)
            self._case = ui.navigation.CaseNavigator(initial_cases, status=self.status, on_change=self._case_changed)
        self._state.observe(self._state_changed)
        self._sync(self._state.selection)
        self._updating = False

    @property
    def case(self) -> ui.navigation.CaseNavigator:
        """Return the case navigator of a view built with ``include_case=True``."""
        if self._case is None:
            message = "This view was built without case navigation."
            raise ValueError(message)
        return self._case

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

    def selected_views(self) -> tuple[selection.EDAView, ...]:
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

    def report_availability(self) -> None:
        """Show the compatibility message and clear it once it no longer applies; other status text stays."""
        message = self.availability_message()
        if message or self._availability_shown:
            self.status.set(message, "warning" if message else "normal")
        self._availability_shown = bool(message)
        if not self._include_case:
            ui.controls.set_visible(self.status.widget, bool(message))

    def deactivate(self) -> None:
        """Ignore shared-state changes until the next activation."""
        self._active = False

    def activate(self) -> None:
        """Synchronize and render this child view from the current global state."""
        self._active = True
        self._sync(self._state.selection)
        if self._callback is not None:
            self._callback()

    def _shared_case_numbers(self) -> tuple[int, ...]:
        """Return exact shared cases for datasets compatible with this child view."""
        return self._state.shared_case_numbers(required_capabilities=self._required_capabilities)

    def _sync_case(self, current: selection.GeneratedOutputSelection) -> None:
        """Show the exact compatible case positions with the selected case chosen."""
        shared = self._shared_case_numbers()
        if self._case is None or not shared:
            return
        selected = shared[0] if current.case_number is None or current.case_number not in shared else current.case_number
        # An unchanged navigator is left alone, so the warning of a corrected typed case stays on the status line.
        if self._case.case_ids != shared:
            self._case.rebind(shared, preserve=False, notify=False)
        if self._case.value != selected:
            self._case.step(shared.index(selected) - self._case.position)

    def _sync(self, current: selection.GeneratedOutputSelection) -> None:
        """Reflect one state update without widget feedback loops."""
        previous = self._updating
        self._updating = True
        try:
            self._sync_case(current)
            self.report_availability()
        finally:
            self._updating = previous

    def _state_changed(self, current: selection.GeneratedOutputSelection) -> None:
        """Synchronize and rerender after any global dataset or case update while active."""
        if not self._active:
            return
        self._sync(current)
        if self._callback is not None:
            self._callback()

    def _case_changed(self, number: int) -> None:
        """Publish one case the navigator accepted to the shared state."""
        if not self._updating:
            self._state.select_case(number, required_capabilities=self._required_capabilities)
