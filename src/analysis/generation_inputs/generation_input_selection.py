"""
===============================================================================
generation_input_selection.py
===============================================================================
Own canonical shared A/B selection state for generation-input analysis.
Responsibilities:
  - Store immutable Dataset A/B and Case A/B canonical keys
  - Reconcile dataset changes against admitted sparse case inventories
  - Notify view-local controls through one guarded session-scoped state owner
  - Resolve deterministic admitted-catalog defaults and optional preferences
Design principles:
  - Canonical keys remain authoritative over labels, paths, and widget positions
  - Dataset changes retain valid case numbers before deterministic fallback
  - State remains in memory and never becomes a durable user preference
This module does NOT:
  - Construct widgets, discover storage, or render scientific diagnostics
  - Modify completed-output EDA or evaluation selection behavior
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import generation_input_sources as sources

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class GenerationInputSelection:
    """Hold one complete canonical generation-input A/B selection."""

    dataset_a_key: sources.DatasetKey
    case_a_key: sources.CaseKey
    dataset_b_key: sources.DatasetKey
    case_b_key: sources.CaseKey


@dataclass(frozen=True, slots=True)
class GenerationInputSelectionResolution:
    """Return admitted defaults and concise fallback issues."""

    selection: GenerationInputSelection
    issues: tuple[str, ...]


def _case_keys(
    catalog: sources.GenerationInputDatasetCatalog,
    dataset_key: sources.DatasetKey,
) -> tuple[sources.CaseKey, ...]:
    """Return admitted case keys in canonical dataset order."""
    return tuple(key for _label, key in catalog.case_options(dataset_key))


def _case_index(
    catalog: sources.GenerationInputDatasetCatalog,
    case_key: sources.CaseKey,
) -> int:
    """Return the canonical numeric index behind one case key."""
    return catalog.reference(case_key).case_index


def _case_for_index(
    catalog: sources.GenerationInputDatasetCatalog,
    dataset_key: sources.DatasetKey,
    case_index: int,
) -> sources.CaseKey | None:
    """Resolve one admitted case index without assuming dense numbering."""
    for case_key in _case_keys(catalog, dataset_key):
        if _case_index(catalog, case_key) == case_index:
            return case_key
    return None


def _first_case(
    catalog: sources.GenerationInputDatasetCatalog,
    dataset_key: sources.DatasetKey,
) -> sources.CaseKey:
    """Return the first admitted case for one validated dataset."""
    cases = _case_keys(catalog, dataset_key)
    if not cases:
        message = f"Generation-input dataset has no admitted cases: {dataset_key!r}."
        raise ValueError(message)
    return cases[0]


def _next_case(
    catalog: sources.GenerationInputDatasetCatalog,
    dataset_key: sources.DatasetKey,
    case_key: sources.CaseKey,
) -> sources.CaseKey:
    """Return the next admitted case after a key, or the key itself."""
    cases = _case_keys(catalog, dataset_key)
    try:
        position = cases.index(case_key)
    except ValueError as error:
        message = f"Case does not belong to generation-input dataset {dataset_key!r}."
        raise ValueError(message) from error
    return cases[position + 1] if position + 1 < len(cases) else case_key


def resolve_generation_input_selection(
    catalog: sources.GenerationInputDatasetCatalog,
    *,
    preferred_dataset_key: sources.DatasetKey | None = None,
    preferred_case_index: int | None = None,
) -> GenerationInputSelectionResolution:
    """
    Resolve optional preferences to one valid admitted A/B selection.

    The first admitted dataset is the deterministic fallback. Case B is the
    next admitted case after Case A, including for sparse case numbering.
    """
    if not isinstance(catalog, sources.GenerationInputDatasetCatalog):
        message = "Generation-input selection requires a GenerationInputDatasetCatalog."
        raise TypeError(message)
    dataset_keys = tuple(sources.dataset_key(dataset) for dataset in catalog.datasets)
    if not dataset_keys:
        message = "Generation-input selection requires at least one admitted dataset."
        raise ValueError(message)
    issues: list[str] = []
    if preferred_dataset_key in dataset_keys:
        dataset_key = preferred_dataset_key
    else:
        compatible_keys = (
            tuple(key for key in dataset_keys if preferred_dataset_key is not None and key[0] == preferred_dataset_key[0])
            if preferred_dataset_key is not None
            else dataset_keys
        )
        dataset_key = (compatible_keys or dataset_keys)[0]
        if preferred_dataset_key is not None:
            fallback = (
                "the first admitted compatible dataset"
                if compatible_keys
                else ("the first admitted dataset because no same-profile dataset is available")
            )
            issues.append(f"Configured batch is the preferred EDA dataset but is not currently admitted; using {fallback}.")
    case_a = None
    if preferred_case_index is not None:
        if isinstance(preferred_case_index, bool) or not isinstance(preferred_case_index, int):
            message = "preferred_case_index must be an integer when supplied."
            raise TypeError(message)
        case_a = _case_for_index(catalog, dataset_key, preferred_case_index)
        if case_a is None:
            issues.append(f"Configured starting case {preferred_case_index} is not admitted; using the first admitted case.")
    if case_a is None:
        case_a = _first_case(catalog, dataset_key)
    return GenerationInputSelectionResolution(
        selection=GenerationInputSelection(
            dataset_a_key=dataset_key,
            case_a_key=case_a,
            dataset_b_key=dataset_key,
            case_b_key=_next_case(catalog, dataset_key, case_a),
        ),
        issues=tuple(issues),
    )


class GenerationInputSelectionState:
    """Own one guarded session-scoped canonical A/B selection."""

    def __init__(
        self,
        catalog: sources.GenerationInputDatasetCatalog,
        *,
        initial_selection: GenerationInputSelection | None = None,
    ) -> None:
        """Validate and retain one authoritative selection owner."""
        if not isinstance(catalog, sources.GenerationInputDatasetCatalog):
            message = "Generation-input state requires a GenerationInputDatasetCatalog."
            raise TypeError(message)
        self._catalog = catalog
        self._observers: list[Callable[[GenerationInputSelection], None]] = []
        self._notifying = False
        resolved = resolve_generation_input_selection(catalog).selection if initial_selection is None else initial_selection
        self._validate(resolved)
        self._selection = resolved

    @property
    def catalog(self) -> sources.GenerationInputDatasetCatalog:
        """Return the immutable catalog bound to this state owner."""
        return self._catalog

    @property
    def selection(self) -> GenerationInputSelection:
        """Return the current immutable canonical selection."""
        return self._selection

    def observe(
        self,
        callback: Callable[[GenerationInputSelection], None],
    ) -> None:
        """Register one control synchronizer exactly once."""
        if callback not in self._observers:
            self._observers.append(callback)

    def _validate(self, value: GenerationInputSelection) -> None:
        """Require complete compatible dataset and case membership."""
        if not isinstance(value, GenerationInputSelection):
            message = "initial_selection must be a GenerationInputSelection."
            raise TypeError(message)
        dataset_a = self._catalog.dataset(value.dataset_a_key)
        dataset_b = self._catalog.dataset(value.dataset_b_key)
        if dataset_a.profile_id != dataset_b.profile_id:
            message = "Generation-input A/B datasets must use one simulation profile."
            raise ValueError(message)
        for dataset_key, case_key, label in (
            (value.dataset_a_key, value.case_a_key, "Case A"),
            (value.dataset_b_key, value.case_b_key, "Case B"),
        ):
            reference = self._catalog.reference(case_key)
            if case_key[:2] != dataset_key or reference.profile_id != dataset_key[0]:
                message = f"{label} does not belong to its selected dataset."
                raise ValueError(message)

    def _replace(self, value: GenerationInputSelection) -> None:
        """Install one changed selection and notify controls without recursion."""
        self._validate(value)
        if value == self._selection:
            return
        self._selection = value
        if self._notifying:
            return
        self._notifying = True
        try:
            for callback in tuple(self._observers):
                callback(value)
        finally:
            self._notifying = False

    def select_dataset_a(self, dataset_key: sources.DatasetKey) -> None:
        """Select Dataset A, retaining its case number when admitted."""
        dataset = self._catalog.dataset(dataset_key)
        current = self._selection
        prior_index = _case_index(self._catalog, current.case_a_key)
        case_a = _case_for_index(self._catalog, dataset_key, prior_index) or _first_case(
            self._catalog,
            dataset_key,
        )
        dataset_b_key = current.dataset_b_key
        case_b = current.case_b_key
        if self._catalog.dataset(dataset_b_key).profile_id != dataset.profile_id:
            dataset_b_key = dataset_key
            case_b = _next_case(self._catalog, dataset_key, case_a)
        self._replace(GenerationInputSelection(dataset_key, case_a, dataset_b_key, case_b))

    def select_dataset_b(self, dataset_key: sources.DatasetKey) -> None:
        """Select compatible Dataset B with deterministic case reconciliation."""
        current = self._selection
        if self._catalog.dataset(dataset_key).profile_id != current.dataset_a_key[0]:
            message = "Dataset B must use Dataset A's simulation profile."
            raise ValueError(message)
        prior_index = _case_index(self._catalog, current.case_b_key)
        case_b = _case_for_index(self._catalog, dataset_key, prior_index)
        if case_b is None:
            case_b = (
                _next_case(self._catalog, dataset_key, current.case_a_key)
                if dataset_key == current.dataset_a_key
                else _first_case(self._catalog, dataset_key)
            )
        self._replace(
            GenerationInputSelection(
                current.dataset_a_key,
                current.case_a_key,
                dataset_key,
                case_b,
            )
        )

    def select_case_a(self, case_key: sources.CaseKey) -> None:
        """Select one admitted Case A without changing independent Case B."""
        current = self._selection
        self._replace(
            GenerationInputSelection(
                current.dataset_a_key,
                case_key,
                current.dataset_b_key,
                current.case_b_key,
            )
        )

    def select_case_b(self, case_key: sources.CaseKey) -> None:
        """Select one admitted Case B without changing independent Case A."""
        current = self._selection
        self._replace(
            GenerationInputSelection(
                current.dataset_a_key,
                current.case_a_key,
                current.dataset_b_key,
                case_key,
            )
        )
