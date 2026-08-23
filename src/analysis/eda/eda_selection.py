"""
eda_selection.py

Own lazy generated-output EDA views and one shared adaptive selection state.

Responsibilities:
  - Describe one admitted batch through its authoritative simulation profile
  - Materialize and cache validated dataframes only when compatible views need them
  - Retain one session-scoped dataset set and exact shared case selection
  - Resolve profile-derived scientific capabilities without user task selection

Design principles:
  - Generation admission evidence remains authoritative for case identity
  - Simulation profiles describe scientific capability; they are not UI filters
  - Widgets mirror this state owner rather than defining scientific selection

This module does NOT:
  - Discover storage, validate Generation publications, or construct widgets
  - Implement plots, time alignment, or Dataset and Training admission
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import pandas as pd

from src import generation

from . import eda_sources as sources

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

GeneratedOutputViewKey: TypeAlias = str
GeneratedOutputCapability: TypeAlias = str

_CAPABILITY_GENERATED_OUTPUT: GeneratedOutputCapability = "generated_output"
_CAPABILITY_SPATIAL_FIELDS: GeneratedOutputCapability = "spatial_fields"
_CAPABILITY_SPECTRAL: GeneratedOutputCapability = "spectral"
_CAPABILITY_TRANSIENT_STATE: GeneratedOutputCapability = "transient_state"
_CAPABILITY_PHYSICAL_TIME: GeneratedOutputCapability = "physical_time"
_CAPABILITY_SCHEDULE: GeneratedOutputCapability = "schedule"
_CAPABILITY_COMPLETION: GeneratedOutputCapability = "completion"


def _profile_capabilities(simulation_profile: str) -> frozenset[GeneratedOutputCapability]:
    """Derive stable EDA capabilities from one authoritative profile schema."""
    profile = generation.contracts.profiles.resolve_profile(simulation_profile)
    capabilities: set[GeneratedOutputCapability] = {
        _CAPABILITY_GENERATED_OUTPUT,
        _CAPABILITY_SPATIAL_FIELDS,
        _CAPABILITY_SPECTRAL,
    }
    learning_views = set(profile.available_learning_views)
    transient_view = generation.contracts.profiles.TRANSIENT_DRYING_LEARNING_VIEW
    if transient_view in learning_views:
        capabilities.update(
            {
                _CAPABILITY_TRANSIENT_STATE,
                _CAPABILITY_PHYSICAL_TIME,
                _CAPABILITY_SCHEDULE,
                _CAPABILITY_COMPLETION,
            }
        )
    return frozenset(capabilities)


class GeneratedOutputEDAView:
    """Describe and lazily materialize one admitted generated-output batch."""

    def __init__(
        self,
        *,
        label: str,
        batch: sources.GeneratedOutputEDABatch,
        case_limit: int | None,
        loader: Callable[[], pd.DataFrame],
    ) -> None:
        """Retain lightweight identity, profile metadata, and one dataframe loader."""
        if not isinstance(label, str) or not label.strip():
            message = "Generated-output view labels must be non-empty text."
            raise ValueError(message)
        if case_limit is not None and (isinstance(case_limit, bool) or not isinstance(case_limit, int) or case_limit <= 0):
            message = "Generated-output view case_limit must be positive or None."
            raise ValueError(message)
        selected = batch.cases if case_limit is None else batch.cases[:case_limit]
        if not selected:
            message = f"Generated-output view {batch.batch_id!r} requires an admitted case."
            raise ValueError(message)
        self._label = label.strip()
        self._batch = batch
        self._cases = tuple(selected)
        self._loader = loader
        self._capabilities = _profile_capabilities(batch.simulation_profile)
        self._frame: pd.DataFrame | None = None

    @property
    def key(self) -> GeneratedOutputViewKey:
        """Return the canonical in-memory key for this admitted batch."""
        return self._batch.batch_id

    @property
    def simulation_profile(self) -> str:
        """Return the authoritative internal simulation-profile identity."""
        return self._batch.simulation_profile

    @property
    def task_id(self) -> str:
        """Return the internal task identity used by the profile-native loader."""
        return self._batch.simulation_profile

    @property
    def capabilities(self) -> frozenset[GeneratedOutputCapability]:
        """Return declared scientific capabilities without loading scientific arrays."""
        return self._capabilities

    def supports(self, required_capabilities: Iterable[GeneratedOutputCapability]) -> bool:
        """Return whether this profile supplies every requested capability."""
        required = frozenset(required_capabilities)
        return required.issubset(self._capabilities)

    @property
    def label(self) -> str:
        """Return the concise globally unique dataset label."""
        return self._label

    @property
    def batch(self) -> sources.GeneratedOutputEDABatch:
        """Return the authoritative admitted Generation batch evidence."""
        return self._batch

    @property
    def case_numbers(self) -> tuple[int, ...]:
        """Return bounded admitted case numbers in canonical order."""
        return tuple(case.case_index for case in self._cases)

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return bounded admitted case identifiers in canonical order."""
        return tuple(case.case_id for case in self._cases)

    @property
    def case_count(self) -> int:
        """Return the number of case references available to this view."""
        return len(self._cases)

    @property
    def is_loaded(self) -> bool:
        """Return whether scientific payloads have been materialized."""
        return self._frame is not None

    def case_id(self, case_number: int) -> str:
        """Resolve one exact admitted case number to its canonical identifier."""
        matches = tuple(case.case_id for case in self._cases if case.case_index == case_number)
        if len(matches) != 1:
            message = f"Generated-output view {self.key!r} has no unique case number {case_number}."
            raise ValueError(message)
        return matches[0]

    def load(self) -> pd.DataFrame:
        """Load, validate, and cache this profile-native dataframe once."""
        if self._frame is None:
            frame = self._loader()
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                message = f"Generated-output loader returned no cases for {self.key!r}."
                raise ValueError(message)
            if frame.attrs.get("task_id") != self.simulation_profile:
                message = f"Generated-output dataframe profile disagrees with {self.key!r}."
                raise ValueError(message)
            expected = self.case_ids
            observed = tuple(str(value) for value in frame.index)
            if observed != expected:
                message = f"Generated-output dataframe case membership disagrees with admitted view {self.key!r}."
                raise ValueError(message)
            self._frame = frame
        return self._frame


class GeneratedOutputEDACatalog:
    """Hold deterministic profile-native views over one admitted source catalog."""

    def __init__(
        self,
        views: Iterable[GeneratedOutputEDAView],
        *,
        source_catalog: sources.GeneratedOutputEDACatalog,
    ) -> None:
        """Validate globally unique batch keys and concise labels without loading frames."""
        resolved = tuple(views)
        if not isinstance(source_catalog, sources.GeneratedOutputEDACatalog):
            message = "Generated-output catalog requires its source admission catalog."
            raise TypeError(message)
        keys = tuple(view.key for view in resolved)
        if len(keys) != len(set(keys)):
            message = "Generated-output catalog view keys must be globally unique."
            raise ValueError(message)
        labels = tuple(view.label for view in resolved)
        if len(labels) != len(set(labels)):
            message = "Generated-output labels must be globally unique."
            raise ValueError(message)
        self._views = resolved
        self._source_catalog = source_catalog
        self._views_by_key = {view.key: view for view in resolved}

    @property
    def views(self) -> tuple[GeneratedOutputEDAView, ...]:
        """Return profile-native views in deterministic presentation order."""
        return self._views

    @property
    def source_catalog(self) -> sources.GeneratedOutputEDACatalog:
        """Return the authoritative source-level admission accounting."""
        return self._source_catalog

    @property
    def capabilities(self) -> frozenset[GeneratedOutputCapability]:
        """Return the union of capabilities declared across every catalog view."""
        return frozenset().union(*(view.capabilities for view in self._views))

    def view(self, key: GeneratedOutputViewKey) -> GeneratedOutputEDAView:
        """Resolve one unique profile-native view."""
        try:
            return self._views_by_key[key]
        except KeyError as error:
            message = f"Generated-output EDA view is unavailable: {key!r}."
            raise ValueError(message) from error

    def views_for_capabilities(
        self,
        required_capabilities: Iterable[GeneratedOutputCapability],
        *,
        keys: Iterable[GeneratedOutputViewKey] | None = None,
    ) -> tuple[GeneratedOutputEDAView, ...]:
        """Return selected or catalog views supplying every requested capability."""
        candidates = self._views if keys is None else tuple(self.view(key) for key in keys)
        return tuple(view for view in candidates if view.supports(required_capabilities))

    def labels(self) -> Mapping[GeneratedOutputViewKey, str]:
        """Return stable globally unique view-key to concise-label mappings."""
        return {view.key: view.label for view in self._views}

    def frames(
        self,
        keys: Iterable[GeneratedOutputViewKey],
        *,
        required_capabilities: Iterable[GeneratedOutputCapability] = (),
    ) -> dict[str, pd.DataFrame]:
        """Load only selected compatible views and return label-keyed frames."""
        selected = tuple(keys)
        if not selected:
            return {}
        compatible = self.views_for_capabilities(required_capabilities, keys=selected)
        return {view.label: view.load() for view in compatible}


@dataclass(frozen=True, slots=True)
class GeneratedOutputSelection:
    """Hold the one current dataset set and exact shared case number."""

    dataset_keys: tuple[GeneratedOutputViewKey, ...]
    case_number: int | None


def _shared_case_numbers(
    catalog: GeneratedOutputEDACatalog,
    keys: tuple[GeneratedOutputViewKey, ...],
) -> tuple[int, ...]:
    """Return exact case-number intersection in the first selected view order."""
    if not keys:
        return ()
    views = tuple(catalog.view(key) for key in keys)
    shared = set(views[0].case_numbers)
    for view in views[1:]:
        shared.intersection_update(view.case_numbers)
    return tuple(number for number in views[0].case_numbers if number in shared)


class GeneratedOutputSelectionState:
    """Own one guarded global selection across adaptive EDA views."""

    def __init__(self, catalog: GeneratedOutputEDACatalog) -> None:
        """Resolve the all-dataset default without materializing scientific payloads."""
        if not isinstance(catalog, GeneratedOutputEDACatalog):
            message = "Generated-output selection requires its view catalog."
            raise TypeError(message)
        if not catalog.views:
            message = "Generated-output selection requires at least one view."
            raise ValueError(message)
        self._catalog = catalog
        self._observers: list[Callable[[GeneratedOutputSelection], None]] = []
        self._notifying = False
        self._channel_preferences: dict[str, tuple[str, ...]] = {}
        self._scope_preferences: dict[str, str] = {}
        self._physical_time_preferences: dict[str, float] = {}
        self._scale_preferences: dict[str, str] = {}
        keys = tuple(view.key for view in catalog.views)
        shared = _shared_case_numbers(catalog, keys)
        self._selection = GeneratedOutputSelection(
            dataset_keys=keys,
            case_number=shared[0] if shared else None,
        )

    @property
    def catalog(self) -> GeneratedOutputEDACatalog:
        """Return the immutable view catalog bound to this state owner."""
        return self._catalog

    @property
    def selection(self) -> GeneratedOutputSelection:
        """Return the one current global selection."""
        return self._selection

    @staticmethod
    def _preference_key(capability: str) -> str:
        """Validate one explicit in-memory view-capability preference key."""
        if not isinstance(capability, str):
            message = "EDA preference capability keys must be text."
            raise TypeError(message)
        key = capability.strip()
        if not key:
            message = "EDA preference capability keys must be non-empty."
            raise ValueError(message)
        return key

    def channel_selection(self, capability: str) -> tuple[str, ...] | None:
        """Return the retained ordered channel selection for one capability."""
        return self._channel_preferences.get(self._preference_key(capability))

    def select_channels(self, capability: str, channels: Iterable[str]) -> None:
        """Retain one unique ordered channel selection without altering identity."""
        key = self._preference_key(capability)
        resolved = tuple(channels)
        if len(resolved) != len(set(resolved)) or any(not isinstance(channel, str) or not channel for channel in resolved):
            message = "EDA channel preferences require unique non-empty names."
            raise ValueError(message)
        self._channel_preferences[key] = resolved

    def scope_selection(self, capability: str, *, default: str) -> str:
        """Return one retained aggregate/single scope."""
        key = self._preference_key(capability)
        if default not in {"aggregate", "single"}:
            message = "EDA scope defaults must be 'aggregate' or 'single'."
            raise ValueError(message)
        return self._scope_preferences.get(key, default)

    def select_scope(self, capability: str, scope: str) -> None:
        """Retain one aggregate/single scope for compatible views."""
        key = self._preference_key(capability)
        if scope not in {"aggregate", "single"}:
            message = "EDA scope preferences must be 'aggregate' or 'single'."
            raise ValueError(message)
        self._scope_preferences[key] = scope

    def physical_time_selection(self, capability: str) -> float | None:
        """Return one retained exact physical time for a view capability."""
        return self._physical_time_preferences.get(self._preference_key(capability))

    def select_physical_time(self, capability: str, physical_time: float) -> None:
        """Retain one finite selected physical time in canonical hours."""
        key = self._preference_key(capability)
        value = float(physical_time)
        if not math.isfinite(value):
            message = "EDA physical-time preferences must be finite."
            raise ValueError(message)
        self._physical_time_preferences[key] = value

    def scale_selection(self, capability: str, *, default: str = "shared") -> str:
        """Return one retained shared/individual color-scale mode."""
        key = self._preference_key(capability)
        if default not in {"shared", "individual"}:
            message = "EDA scale defaults must be 'shared' or 'individual'."
            raise ValueError(message)
        return self._scale_preferences.get(key, default)

    def select_scale(self, capability: str, scale: str) -> None:
        """Retain one shared/individual color-scale mode."""
        key = self._preference_key(capability)
        if scale not in {"shared", "individual"}:
            message = "EDA scale preferences must be 'shared' or 'individual'."
            raise ValueError(message)
        self._scale_preferences[key] = scale

    def shared_case_numbers(
        self,
        *,
        required_capabilities: Iterable[GeneratedOutputCapability] = (),
    ) -> tuple[int, ...]:
        """Return exact shared cases across selected views compatible with one view."""
        compatible = self._catalog.views_for_capabilities(
            required_capabilities,
            keys=self._selection.dataset_keys,
        )
        return _shared_case_numbers(self._catalog, tuple(view.key for view in compatible))

    def selected_views(
        self,
        *,
        required_capabilities: Iterable[GeneratedOutputCapability] = (),
    ) -> tuple[GeneratedOutputEDAView, ...]:
        """Return selected views compatible with every requested capability."""
        return self._catalog.views_for_capabilities(
            required_capabilities,
            keys=self._selection.dataset_keys,
        )

    def observe(self, callback: Callable[[GeneratedOutputSelection], None]) -> None:
        """Register one state observer exactly once."""
        if callback not in self._observers:
            self._observers.append(callback)

    def _notify(self) -> None:
        """Notify every observer once without recursive state replacement."""
        if self._notifying:
            return
        self._notifying = True
        try:
            for callback in tuple(self._observers):
                callback(self._selection)
        finally:
            self._notifying = False

    def select_datasets(self, keys: Iterable[GeneratedOutputViewKey]) -> None:
        """Select any ordered dataset subset and reconcile its shared case exactly."""
        resolved = tuple(keys)
        if len(resolved) != len(set(resolved)):
            message = "Generated-output dataset selection cannot contain duplicates."
            raise ValueError(message)
        available_order = tuple(view.key for view in self._catalog.views)
        if any(key not in available_order for key in resolved):
            message = "Generated-output dataset selection contains an unavailable view."
            raise ValueError(message)
        requested = set(resolved)
        ordered = tuple(key for key in available_order if key in requested)
        shared = _shared_case_numbers(self._catalog, ordered)
        current = self._selection
        case_number = current.case_number if current.case_number in shared else (shared[0] if shared else None)
        replacement = GeneratedOutputSelection(dataset_keys=ordered, case_number=case_number)
        if replacement == current:
            return
        self._selection = replacement
        self._notify()

    def select_case(
        self,
        case_number: int,
        *,
        required_capabilities: Iterable[GeneratedOutputCapability] = (),
    ) -> None:
        """Select one case shared by the compatible selected-dataset subset."""
        if isinstance(case_number, bool) or not isinstance(case_number, int):
            message = "Generated-output case numbers must be integers."
            raise TypeError(message)
        shared = self.shared_case_numbers(required_capabilities=required_capabilities)
        if case_number not in shared:
            message = f"Case {case_number} is unavailable for the compatible selected datasets."
            raise ValueError(message)
        current = self._selection
        replacement = GeneratedOutputSelection(dataset_keys=current.dataset_keys, case_number=case_number)
        if replacement == current:
            return
        self._selection = replacement
        self._notify()
