"""
evaluation_selection.py

Own synchronized Evaluation notebook selection state.

Responsibilities:
  - Retain selected experiments, child runs, artifact role, and analysis scope
  - Bind exact artifact channels, cases, physical times, protocols, and horizons
  - Notify dependent views with the precise fields changed
  - Reject task-incompatible or artifact-unsupported selections

Design principles:
  - Persisted run and artifact identities remain separate from display labels
  - Steady and transient controls are admitted only from declared capabilities
  - State changes are immutable snapshots with one observable owner

This module does NOT:
  - Discover runs, load artifacts, build widgets, or render plots
  - Infer physical times, rollout horizons, or channels from filenames
  - Trigger model inference or artifact generation
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from src.analysis.evaluation import evaluation_run_discovery as run_discovery

EvaluationScope = Literal["aggregate", "single"]
ArtifactRole = Literal["id", "ood", "both"]
EvaluationProtocol = Literal[
    "teacher_forced_one_step",
    "autonomous_full",
    "rolling_origin",
]
SelectionObserver = Callable[
    ["EvaluationSelection", "EvaluationSelection", frozenset[str]],
    None,
]

_SCOPE_VALUES = frozenset({"aggregate", "single"})
_ARTIFACT_ROLES = frozenset({"id", "ood", "both"})
_TRANSIENT_PROTOCOLS = frozenset(
    {
        "teacher_forced_one_step",
        "autonomous_full",
        "rolling_origin",
    }
)


@dataclass(frozen=True, slots=True)
class EvaluationViewCapabilities:
    """Describe exact interactive coordinates admitted from selected artifacts."""

    task: str
    channels: tuple[str, ...]
    case_ids: tuple[str, ...]
    physical_times: tuple[float, ...] = ()
    protocols: tuple[EvaluationProtocol, ...] = ()
    horizons: tuple[int | str, ...] = ()

    def __post_init__(self) -> None:
        """Validate one task-owned capability inventory."""
        if self.task not in {"steady_flow", "transient_drying"}:
            message = f"Unsupported Evaluation task {self.task!r}."
            raise ValueError(message)
        if not self.channels or len(self.channels) != len(set(self.channels)):
            message = "Evaluation channels must be unique and non-empty."
            raise ValueError(message)
        if len(self.case_ids) != len(set(self.case_ids)):
            message = "Evaluation case identities must be unique."
            raise ValueError(message)
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) for value in self.physical_times):
            message = "Physical-time capabilities must contain finite real coordinates."
            raise TypeError(message)
        times = tuple(float(value) for value in self.physical_times)
        if times and any(right <= left for left, right in pairwise(times)):
            message = "Physical-time capabilities must be strictly increasing."
            raise ValueError(message)
        if len(self.protocols) != len(set(self.protocols)) or set(self.protocols).difference(_TRANSIENT_PROTOCOLS):
            message = "Evaluation protocols contain duplicates or unknown values."
            raise ValueError(message)
        if len(self.horizons) != len(set(self.horizons)):
            message = "Rollout horizons must be unique."
            raise ValueError(message)
        if self.task == "steady_flow" and (self.physical_times or self.protocols or self.horizons):
            message = "Steady Evaluation cannot expose transient time or horizon controls."
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class EvaluationSelection:
    """Store one immutable synchronized notebook selection."""

    task: str
    experiment_identities: tuple[str, ...]
    run_dirs: tuple[Path, ...]
    artifact_role: ArtifactRole
    scope: EvaluationScope
    channels: tuple[str, ...]
    case_id: str | None
    physical_time: float | None
    protocol: EvaluationProtocol | None
    horizon: int | str | None


class EvaluationSelectionState:
    """Coordinate run-level and artifact-level Evaluation choices."""

    def __init__(
        self,
        catalog: run_discovery.EvaluationRunCatalog,
        *,
        comparison: bool,
    ) -> None:
        """Select the first deterministic experiment without loading artifacts."""
        if not isinstance(catalog, run_discovery.EvaluationRunCatalog):
            message = "Evaluation selection requires an EvaluationRunCatalog."
            raise TypeError(message)
        if not catalog.groups:
            message = "Evaluation selection requires at least one discovered experiment."
            raise ValueError(message)
        group = catalog.groups[0]
        selected_children = tuple(child.run_dir for child in group.children) if comparison else (group.children[0].run_dir,)
        self._catalog = catalog
        self._comparison = comparison
        self._capabilities: EvaluationViewCapabilities | None = None
        self._observers: list[SelectionObserver] = []
        self._selection = EvaluationSelection(
            task=group.task,
            experiment_identities=(group.identity_sha256,),
            run_dirs=selected_children,
            artifact_role="id",
            scope="aggregate",
            channels=(),
            case_id=None,
            physical_time=None,
            protocol=None,
            horizon=None,
        )

    @property
    def catalog(self) -> run_discovery.EvaluationRunCatalog:
        """Return the immutable discovery catalog bound to this state."""
        return self._catalog

    @property
    def comparison(self) -> bool:
        """Return whether multiple compatible runs may be selected."""
        return self._comparison

    @property
    def selection(self) -> EvaluationSelection:
        """Return the current immutable selection snapshot."""
        return self._selection

    @property
    def capabilities(self) -> EvaluationViewCapabilities | None:
        """Return current artifact capabilities, if a role has been admitted."""
        return self._capabilities

    def observe(self, callback: SelectionObserver) -> None:
        """Register one unique state observer."""
        if not callable(callback):
            message = "Evaluation selection observers must be callable."
            raise TypeError(message)
        if callback not in self._observers:
            self._observers.append(callback)

    def unobserve(self, callback: SelectionObserver) -> None:
        """Remove one observer when present."""
        if callback in self._observers:
            self._observers.remove(callback)

    def groups_by_task(
        self,
    ) -> Mapping[str, tuple[run_discovery.EvaluationRunGroup, ...]]:
        """Return deterministic experiment groups partitioned by persisted task."""
        grouped: dict[str, list[run_discovery.EvaluationRunGroup]] = {}
        for group in self._catalog.groups:
            grouped.setdefault(group.task, []).append(group)
        return MappingProxyType({task: tuple(groups) for task, groups in grouped.items()})

    def group(self) -> run_discovery.EvaluationRunGroup:
        """Return the exact selected experiment group."""
        if len(self._selection.experiment_identities) != 1:
            message = "The current comparison selection spans multiple experiments."
            raise RuntimeError(message)
        identity = self._selection.experiment_identities[0]
        for group in self._catalog.groups:
            if group.identity_sha256 == identity:
                return group
        message = "Selected Evaluation experiment is absent from its catalog."
        raise RuntimeError(message)

    def select_experiment(
        self,
        identity_sha256: str,
        *,
        run_dirs: Sequence[Path | str] | None = None,
    ) -> None:
        """Select one persisted experiment and one or more exact children."""
        matching = tuple(group for group in self._catalog.groups if group.identity_sha256 == identity_sha256)
        if len(matching) != 1:
            message = f"Unknown Evaluation experiment identity {identity_sha256!r}."
            raise KeyError(message)
        group = matching[0]
        available = tuple(child.run_dir for child in group.children)
        if run_dirs is None:
            selected = available if self._comparison else available[:1]
        else:
            selected = tuple(Path(path).expanduser().resolve() for path in run_dirs)
        if not selected or any(path not in available for path in selected):
            message = "Selected Evaluation children must belong to one exact experiment."
            raise ValueError(message)
        if not self._comparison and len(selected) != 1:
            message = "Single-model Evaluation selects exactly one child run."
            raise ValueError(message)
        self._capabilities = None
        self._install(
            replace(
                self._selection,
                task=group.task,
                experiment_identities=(group.identity_sha256,),
                run_dirs=selected,
                channels=(),
                case_id=None,
                physical_time=None,
                protocol=None,
                horizon=None,
            )
        )

    def select_runs(self, run_dirs: Sequence[Path | str]) -> None:
        """Select exact children within the current experiment."""
        if len(self._selection.experiment_identities) != 1:
            message = "Use select_catalog_runs for a multi-experiment comparison."
            raise RuntimeError(message)
        self.select_experiment(
            self._selection.experiment_identities[0],
            run_dirs=run_dirs,
        )

    def select_catalog_runs(self, run_dirs: Sequence[Path | str]) -> None:
        """Select compatible exact runs across one or more catalog experiments."""
        if not self._comparison:
            message = "Cross-experiment run selection is comparison-only."
            raise RuntimeError(message)
        selected = tuple(Path(path).expanduser().resolve() for path in run_dirs)
        if not selected or len(selected) != len(set(selected)):
            message = "Comparison runs must be a unique non-empty exact-path sequence."
            raise ValueError(message)
        by_path = {child.run_dir: (group, child) for group in self._catalog.groups for child in group.children}
        if any(path not in by_path for path in selected):
            message = "Comparison runs must all belong to the current discovery catalog."
            raise ValueError(message)
        groups_and_runs = tuple(by_path[path] for path in selected)
        tasks = {child.task for _group, child in groups_and_runs}
        if len(tasks) != 1:
            message = "Comparison runs must share one persisted task."
            raise ValueError(message)
        identities = tuple(dict.fromkeys(group.identity_sha256 for group, _child in groups_and_runs))
        self._capabilities = None
        self._install(
            replace(
                self._selection,
                task=next(iter(tasks)),
                experiment_identities=identities,
                run_dirs=selected,
                channels=(),
                case_id=None,
                physical_time=None,
                protocol=None,
                horizon=None,
            )
        )

    def select_artifact_role(self, role: ArtifactRole) -> None:
        """Select ID, OOD, or paired ID+OOD and clear role-local coordinates."""
        if role not in _ARTIFACT_ROLES:
            message = f"Unknown Evaluation artifact role {role!r}."
            raise ValueError(message)
        self._capabilities = None
        self._install(
            replace(
                self._selection,
                artifact_role=role,
                channels=(),
                case_id=None,
                physical_time=None,
                protocol=None,
                horizon=None,
            )
        )

    def select_scope(self, scope: EvaluationScope) -> None:
        """Select the EDA-compatible Aggregate or Single case mode."""
        if scope not in _SCOPE_VALUES:
            message = f"Unknown Evaluation scope {scope!r}."
            raise ValueError(message)
        self._install(replace(self._selection, scope=scope))

    def bind_capabilities(
        self,
        capabilities: EvaluationViewCapabilities,
    ) -> None:
        """Bind exact artifact coordinates while preserving valid shared choices."""
        if not isinstance(capabilities, EvaluationViewCapabilities):
            message = "Evaluation capabilities require EvaluationViewCapabilities."
            raise TypeError(message)
        if capabilities.task != self._selection.task:
            message = "Artifact capabilities contradict the selected persisted task."
            raise ValueError(message)
        self._capabilities = capabilities
        channels = tuple(channel for channel in capabilities.channels if channel in self._selection.channels)
        self._install(
            replace(
                self._selection,
                channels=channels or capabilities.channels,
                case_id=(
                    self._selection.case_id
                    if self._selection.case_id in capabilities.case_ids
                    else (capabilities.case_ids[0] if capabilities.case_ids else None)
                ),
                physical_time=(
                    self._selection.physical_time
                    if self._selection.physical_time in capabilities.physical_times
                    else (capabilities.physical_times[-1] if capabilities.physical_times else None)
                ),
                protocol=(
                    self._selection.protocol
                    if self._selection.protocol in capabilities.protocols
                    else (capabilities.protocols[0] if capabilities.protocols else None)
                ),
                horizon=(
                    self._selection.horizon
                    if self._selection.horizon in capabilities.horizons
                    else (capabilities.horizons[0] if capabilities.horizons else None)
                ),
            )
        )

    def select_channels(self, channels: Sequence[str]) -> None:
        """Select one or more channels in artifact-declared order."""
        capabilities = self._require_capabilities()
        selected = tuple(channels)
        if not selected or len(selected) != len(set(selected)) or any(channel not in capabilities.channels for channel in selected):
            message = "Selected channels must be a unique non-empty capability subset."
            raise ValueError(message)
        ordered = tuple(channel for channel in capabilities.channels if channel in selected)
        self._install(replace(self._selection, channels=ordered))

    def select_case(self, case_id: str) -> None:
        """Select one exact case identity from the active artifact role."""
        capabilities = self._require_capabilities()
        if case_id not in capabilities.case_ids:
            message = f"Case {case_id!r} is unavailable for the selected artifact."
            raise ValueError(message)
        self._install(replace(self._selection, case_id=case_id))

    def select_physical_time(self, physical_time: float) -> None:
        """Select one exact stored physical time without interpolation."""
        capabilities = self._require_capabilities()
        value = float(physical_time)
        if value not in capabilities.physical_times:
            message = f"Physical time {value:g} is unavailable for the selected artifact."
            raise ValueError(message)
        self._install(replace(self._selection, physical_time=value))

    def select_protocol(self, protocol: EvaluationProtocol) -> None:
        """Select one artifact-declared one-step or rollout protocol."""
        capabilities = self._require_capabilities()
        if protocol not in capabilities.protocols:
            message = f"Evaluation protocol {protocol!r} is unavailable."
            raise ValueError(message)
        self._install(replace(self._selection, protocol=protocol))

    def select_horizon(self, horizon: int | str) -> None:
        """Select one exact artifact-provided rollout horizon."""
        capabilities = self._require_capabilities()
        if horizon not in capabilities.horizons:
            message = f"Rollout horizon {horizon!r} is unavailable."
            raise ValueError(message)
        self._install(replace(self._selection, horizon=horizon))

    def _require_capabilities(self) -> EvaluationViewCapabilities:
        """Return current capabilities or reject unbound artifact controls."""
        if self._capabilities is None:
            message = "No Evaluation artifact capabilities are currently bound."
            raise RuntimeError(message)
        return self._capabilities

    def _install(self, current: EvaluationSelection) -> None:
        """Install one snapshot and notify observers with exact dependencies."""
        previous = self._selection
        if current == previous:
            return
        self._selection = current
        changed = frozenset(
            field
            for field in (
                "task",
                "experiment_identities",
                "run_dirs",
                "artifact_role",
                "scope",
                "channels",
                "case_id",
                "physical_time",
                "protocol",
                "horizon",
            )
            if getattr(previous, field) != getattr(current, field)
        )
        for callback in tuple(self._observers):
            callback(previous, current, changed)
