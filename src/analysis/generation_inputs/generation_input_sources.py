"""
===============================================================================
generation_input_sources.py
===============================================================================
Group canonical input-generated batches into scientific datasets.
Responsibilities:
  - Discover manifest-admitted input batches without exposing completed raw
  - Group sources by canonical batch identity and deduplicate exact cases
  - Fail closed when one case-input identity binds conflicting persisted bytes
  - Admit selected cases lazily and cache empirical dataset diagnostics
Design principles:
  - Dataset identity is independent of storage location
  - Dataset means always use every unique discovered case in the dataset
  - Visible labels remain concise while selector values retain canonical keys
This module does NOT:
  - Refresh sources, generate inputs, or inspect completed simulation output
  - Compute scientific formulas, render plots, or resample persisted inputs
===============================================================================
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias

from src import common
from src.generation.cases import generation_cases_admission as admission

from . import generation_input_diagnostics as diagnostics
from . import generation_input_labels as labels

if TYPE_CHECKING:
    from collections.abc import Iterable

DatasetKey: TypeAlias = tuple[str, str]
CaseKey: TypeAlias = tuple[str, str, str]

DEFAULT_MAX_CACHED_CASES: Final = 24
DEFAULT_MAX_CACHED_DATASETS: Final = 2
CASE_KEY_PARTS: Final = 3


@dataclass(frozen=True, slots=True)
class GenerationInputDataset:
    """Describe one canonical generation batch and its unique input cases."""

    profile_id: str
    batch_id: str
    batch_storage_name: str
    batch_identity: str
    material_family: str
    sampling_regime: str
    campaign_purpose: str
    publications: tuple[admission.InputSource, ...]
    cases: tuple[admission.InputCaseReference, ...]


def dataset_key(dataset: GenerationInputDataset) -> DatasetKey:
    """Return the immutable profile-qualified canonical dataset key."""
    return dataset.profile_id, dataset.batch_identity


def case_key(
    dataset: GenerationInputDataset,
    reference: admission.InputCaseReference,
) -> CaseKey:
    """Return one dataset-qualified immutable case-input key."""
    return dataset.profile_id, dataset.batch_identity, reference.case_input_id


def case_display_label(reference: admission.InputCaseReference) -> str:
    """Return the numeric-only visible case selector label."""
    return str(reference.case_index)


def _metadata_fingerprint(
    value: object,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    """Return deterministic parameter-presentation metadata evidence."""
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        msg = "Parameter metadata must be mapping-shaped."
        raise TypeError(msg)
    result = []
    for name, entry in sorted(
        value.items(),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(entry, Mapping):
            msg = "Parameter metadata entries must be mapping-shaped."
            raise TypeError(msg)
        result.append(
            (
                str(name),
                tuple(sorted((str(key), str(item)) for key, item in entry.items())),
            )
        )
    return tuple(result)


def _persisted_case_fingerprint(
    reference: admission.InputCaseReference,
) -> tuple[str, tuple[tuple[str, int, str], ...]]:
    """Hash exact duplicate-case payload and adapter bytes."""
    payload_path = reference.case_directory / "case.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    declared = payload.get("input_files")
    if not isinstance(declared, dict):
        msg = "Duplicate input-generated case lacks declared input-file identities."
        raise TypeError(msg)
    files = []
    for filename, identity in sorted(declared.items()):
        if not isinstance(filename, str) or not isinstance(identity, dict):
            msg = "Duplicate input-generated file identity is malformed."
            raise TypeError(msg)
        source = reference.input_directory / filename
        if not source.is_file():
            msg = f"Duplicate input-generated file is missing: {source}"
            raise ValueError(msg)
        size = source.stat().st_size
        digest = common.serialization.file_sha256(source)
        if identity.get("size_bytes") != size or identity.get("sha256") != digest:
            msg = "Duplicate input-generated adapter bytes disagree with case.json."
            raise ValueError(msg)
        files.append((filename, size, digest))
    return common.serialization.file_sha256(payload_path), tuple(files)


def _assert_duplicate_equivalent(
    first: admission.InputCaseReference,
    second: admission.InputCaseReference,
) -> None:
    """Reject one repeated case-input identity with conflicting evidence."""
    first_metadata = (
        first.profile_id,
        first.batch_id,
        first.batch_storage_name,
        first.batch_identity,
        first.material_family,
        first.sampling_regime,
        first.campaign_purpose,
        first.case_id,
        first.case_index,
        first.case_input_id,
        first.simulation_case_id,
        _metadata_fingerprint(first.parameter_metadata),
    )
    second_metadata = (
        second.profile_id,
        second.batch_id,
        second.batch_storage_name,
        second.batch_identity,
        second.material_family,
        second.sampling_regime,
        second.campaign_purpose,
        second.case_id,
        second.case_index,
        second.case_input_id,
        second.simulation_case_id,
        _metadata_fingerprint(second.parameter_metadata),
    )
    if first_metadata != second_metadata or _persisted_case_fingerprint(first) != _persisted_case_fingerprint(second):
        msg = f"Conflicting input-generated sources bind the same case-input identity {first.case_input_id!r}."
        raise ValueError(msg)


def _build_dataset(
    publications: tuple[admission.InputSource, ...],
) -> GenerationInputDataset:
    """Merge exact input-generated sources into one canonical dataset."""
    first = publications[0]
    contract = (
        first.profile_id,
        first.batch_id,
        first.batch_storage_name,
        first.batch_identity,
        first.material_family,
        first.sampling_regime,
        first.campaign_purpose,
    )
    if any(
        (
            source.profile_id,
            source.batch_id,
            source.batch_storage_name,
            source.batch_identity,
            source.material_family,
            source.sampling_regime,
            source.campaign_purpose,
        )
        != contract
        for source in publications[1:]
    ):
        msg = "Input-generated sources grouped as one dataset disagree on canonical batch metadata."
        raise ValueError(msg)
    unique: dict[str, admission.InputCaseReference] = {}
    indices: dict[int, str] = {}
    for source in publications:
        for reference in source.cases:
            previous = unique.get(reference.case_input_id)
            if previous is not None:
                _assert_duplicate_equivalent(previous, reference)
                continue
            previous_identity = indices.get(reference.case_index)
            if previous_identity is not None and previous_identity != reference.case_input_id:
                msg = "One canonical dataset case index binds conflicting case-input identities."
                raise ValueError(msg)
            indices[reference.case_index] = reference.case_input_id
            unique[reference.case_input_id] = reference
    cases = tuple(
        sorted(
            unique.values(),
            key=lambda reference: (
                reference.case_index,
                reference.case_input_id,
            ),
        )
    )
    if not cases:
        msg = "A generation-input dataset has no input-generated cases."
        raise ValueError(msg)
    return GenerationInputDataset(
        profile_id=contract[0],
        batch_id=contract[1],
        batch_storage_name=contract[2],
        batch_identity=contract[3],
        material_family=contract[4],
        sampling_regime=contract[5],
        campaign_purpose=contract[6],
        publications=publications,
        cases=cases,
    )


class GenerationInputDatasetCatalog:
    """Own immutable input datasets and bounded lazy diagnostic caches."""

    def __init__(
        self,
        discovery: admission.InputSourceDiscovery,
        *,
        max_cached_cases: int = DEFAULT_MAX_CACHED_CASES,
        max_cached_datasets: int = DEFAULT_MAX_CACHED_DATASETS,
    ) -> None:
        """
        Group one manifest-first input discovery into scientific datasets.

        Completed-raw sources are deliberately excluded. Dataset membership is
        never truncated because every empirical mean must use all unique cases.
        """
        if not isinstance(discovery, admission.InputSourceDiscovery):
            msg = "Generation-input catalogs require InputSourceDiscovery evidence."
            raise TypeError(msg)
        for name, value in (
            ("max_cached_cases", max_cached_cases),
            ("max_cached_datasets", max_cached_datasets),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                msg = f"{name} must be a positive integer."
                raise ValueError(msg)
        input_sources = tuple(source for source in discovery.sources if source.source_kind == "input_generated" and source.cases)
        identities = tuple(source.source_id for source in input_sources)
        if len(identities) != len(set(identities)):
            msg = "Input-batch discovery contains duplicate source IDs."
            raise ValueError(msg)
        grouped: OrderedDict[
            DatasetKey,
            list[admission.InputSource],
        ] = OrderedDict()
        for source in input_sources:
            key = (source.profile_id, source.batch_identity)
            grouped.setdefault(key, []).append(source)
        self._datasets = tuple(_build_dataset(tuple(publications)) for publications in grouped.values())
        keys = tuple(dataset_key(dataset) for dataset in self._datasets)
        if len(keys) != len(set(keys)):
            msg = "Generation-input datasets have duplicate canonical identities."
            raise ValueError(msg)
        empty_issues = tuple(
            admission.InputDiscoveryIssue(
                source.source_id,
                source.directory,
                "input-generated batch has no case references",
            )
            for source in discovery.sources
            if source.source_kind == "input_generated" and not source.cases
        )
        self._issues = (*discovery.issues, *empty_issues)
        self._max_cached_cases = max_cached_cases
        self._max_cached_datasets = max_cached_datasets
        self._case_cache: OrderedDict[
            CaseKey,
            diagnostics.GenerationInputDiagnostics,
        ] = OrderedDict()
        self._dataset_cache: OrderedDict[
            DatasetKey,
            diagnostics.DatasetDiagnostics,
        ] = OrderedDict()

    @property
    def datasets(self) -> tuple[GenerationInputDataset, ...]:
        """Return canonical datasets in deterministic discovery order."""
        return self._datasets

    @property
    def issues(self) -> tuple[admission.InputDiscoveryIssue, ...]:
        """Return compact invalid-input evidence from notebook discovery."""
        return self._issues

    @property
    def profiles(self) -> tuple[str, ...]:
        """Return represented maintained profiles in stable dataset order."""
        return tuple(dict.fromkeys(dataset.profile_id for dataset in self._datasets))

    @staticmethod
    def _build_labels(
        datasets: tuple[GenerationInputDataset, ...],
    ) -> dict[DatasetKey, str]:
        """Delegate complete visible-label construction to its presentation owner."""
        display_metadata = tuple(
            labels.DatasetLabelMetadata(
                profile_id=dataset.profile_id,
                material_family=dataset.material_family,
                sampling_regime=dataset.sampling_regime,
                campaign_purpose=dataset.campaign_purpose,
                batch_identity=dataset.batch_identity,
            )
            for dataset in datasets
        )
        display_labels = labels.dataset_display_labels(display_metadata)
        return {
            dataset_key(dataset): label
            for dataset, label in zip(
                datasets,
                display_labels,
                strict=True,
            )
        }

    def dataset(self, key: DatasetKey) -> GenerationInputDataset:
        """Resolve one canonical dataset key exactly."""
        matches = tuple(dataset for dataset in self._datasets if dataset_key(dataset) == key)
        if len(matches) != 1:
            msg = f"Generation-input dataset is unavailable or ambiguous: {key!r}."
            raise ValueError(msg)
        return matches[0]

    def reference(self, key: CaseKey) -> admission.InputCaseReference:
        """Resolve one unique dataset-qualified case reference."""
        if len(key) != CASE_KEY_PARTS:
            msg = "Generation-input case key is malformed."
            raise ValueError(msg)
        dataset = self.dataset((key[0], key[1]))
        matches = tuple(reference for reference in dataset.cases if reference.case_input_id == key[2])
        if len(matches) != 1:
            msg = f"Generation-input case is unavailable or ambiguous: {key!r}."
            raise ValueError(msg)
        return matches[0]

    def dataset_options(
        self,
        *,
        profile_ids: Iterable[str] | None = None,
    ) -> tuple[tuple[str, DatasetKey], ...]:
        """Return concise dataset labels and canonical internal keys."""
        allowed = None if profile_ids is None else frozenset(profile_ids)
        visible = tuple(dataset for dataset in self._datasets if allowed is None or dataset.profile_id in allowed)
        labels = self._build_labels(visible)
        return tuple((labels[dataset_key(dataset)], dataset_key(dataset)) for dataset in visible)

    def case_options(
        self,
        key: DatasetKey,
    ) -> tuple[tuple[str, CaseKey], ...]:
        """Return numeric-only labels for every unique dataset case."""
        dataset = self.dataset(key)
        return tuple(
            (
                case_display_label(reference),
                case_key(dataset, reference),
            )
            for reference in dataset.cases
        )

    def load(self, key: CaseKey) -> diagnostics.GenerationInputDiagnostics:
        """Admit one selected input case and update the bounded LRU cache."""
        cached = self._case_cache.pop(key, None)
        if cached is not None:
            self._case_cache[key] = cached
            return cached
        reference = self.reference(key)
        record = diagnostics.build_case_diagnostics(admission.admit_input_case_reference(reference))
        self._case_cache[key] = record
        while len(self._case_cache) > self._max_cached_cases:
            self._case_cache.popitem(last=False)
        return record

    def load_dataset(
        self,
        key: DatasetKey,
    ) -> tuple[diagnostics.GenerationInputDiagnostics, ...]:
        """Load all unique cases from one dataset without display truncation."""
        dataset = self.dataset(key)
        return tuple(self.load(case_key(dataset, reference)) for reference in dataset.cases)

    def dataset_diagnostics(
        self,
        key: DatasetKey,
    ) -> diagnostics.DatasetDiagnostics:
        """Return cached empirical means over every unique dataset case."""
        cached = self._dataset_cache.pop(key, None)
        if cached is not None:
            self._dataset_cache[key] = cached
            return cached
        summary = diagnostics.build_dataset_diagnostics(self.load_dataset(key))
        self._dataset_cache[key] = summary
        while len(self._dataset_cache) > self._max_cached_datasets:
            self._dataset_cache.popitem(last=False)
        return summary


def discover_generation_input_datasets(
    storage_root: Path | str | None = None,
    *,
    max_cached_cases: int = DEFAULT_MAX_CACHED_CASES,
    max_cached_datasets: int = DEFAULT_MAX_CACHED_DATASETS,
) -> GenerationInputDatasetCatalog:
    """
    Discover every valid input batch once and group canonical datasets.

    Discovery is performed when the notebook cell executes. It never refreshes,
    generates inputs, or admits completed-raw publications.
    """
    root = None if storage_root is None else Path(storage_root).expanduser()
    discovery = admission.discover_input_batches(
        root,
        max_sources=None,
        max_cases_per_source=None,
    )
    return GenerationInputDatasetCatalog(
        discovery,
        max_cached_cases=max_cached_cases,
        max_cached_datasets=max_cached_datasets,
    )
