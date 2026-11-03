"""Group canonical input-generated batches into scientific datasets.

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

Notes:
  Discovery products:
    - GenerationInputDatasetCatalog groups admitted input publications and loads case diagnostics lazily for selected views.
    - CampaignOutcomeCatalog is a separate candidate-lifecycle view joined to admitted input metadata; its rows do not change canonical input-dataset
      membership.

"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, TypeAlias

import numpy as np
import pandas as pd

from src import common
from src.generation.cases import generation_cases_admission as admission
from src.generation.contracts import generation_contracts_profiles as profiles
from src.generation.publication import generation_publication_catalog as campaign_catalog

from . import generation_input_diagnostics as diagnostics
from . import generation_input_labels as labels

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

DatasetKey: TypeAlias = tuple[str, str]  # noqa: UP040 -- public runtime aliases must remain their concrete typing or class objects
CaseKey: TypeAlias = tuple[str, str, str]  # noqa: UP040 -- public runtime aliases must remain their concrete typing or class objects

DEFAULT_MAX_CACHED_CASES: Final = 24
DEFAULT_MAX_CACHED_DATASETS: Final = 2
CASE_KEY_PARTS: Final = 3
_CANDIDATE_COLUMNS: Final = ("material_family", "domain_profile", "case_id", "origin", "outcome", "selected", "design_key", "duration_seconds")
# Leading columns of the campaign outcome table; every sampled parameter of a candidate follows as its own column.
CAMPAIGN_OUTCOME_COLUMNS: Final = (*_CANDIDATE_COLUMNS, "attempts", "failure_stage", "failure_class", "reason_code", "input_admission")


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
        source = reference.case_directory / filename
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
        outcomes: CampaignOutcomeCatalog | None = None,
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
        self.outcomes = outcomes or CampaignOutcomeCatalog(())
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
    root = common.paths.get_storage_root(storage_root=storage_root).expanduser().resolve()
    discovery = admission.discover_input_batches(
        root,
        max_sources=None,
        max_cases_per_source=None,
        validation_depth="evidence",
    )
    return GenerationInputDatasetCatalog(
        discovery,
        max_cached_cases=max_cached_cases,
        max_cached_datasets=max_cached_datasets,
        outcomes=discover_campaign_outcomes(root, discovery),
    )


@dataclass(frozen=True, slots=True)
class CampaignOutcomeCatalog:
    """Retain one bounded discovery checkpoint without payload-array reads."""

    rows: tuple[dict[str, Any], ...]
    issues: tuple[str, ...] = ()


def discover_campaign_outcomes(storage_root: Path, discovery: admission.InputSourceDiscovery) -> CampaignOutcomeCatalog:
    """Join immutable candidate decisions to already admitted input metadata."""
    references = {
        str((reference.case_directory / "case.json").relative_to(storage_root)): reference
        for source in discovery.sources
        for reference in source.cases
        if reference.case_directory.is_relative_to(storage_root)
    }
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for task in profiles.available_profiles():
        try:
            campaigns = campaign_catalog.list_campaigns(task, storage_root=storage_root)
        except (OSError, ValueError, TypeError, KeyError) as error:
            issues.append(f"{task}: {error}")
            continue
        for campaign in campaigns:
            name = campaign["source_name"]
            try:
                candidates = campaign_catalog.list_campaign_cases(task, name, storage_root=storage_root)
            except (OSError, ValueError, TypeError, KeyError) as error:
                issues.append(f"{task}/{name}: {error}")
                continue
            for candidate in candidates:
                row = dict(candidate)
                input_locator = row.get("input_case_relative")
                reference = references.get(input_locator) if isinstance(input_locator, str) else None
                if reference is not None:
                    payload = reference.case_payload()
                    row["parameters"] = payload["sampled_values"]
                    row["parameter_units"] = payload["sampled_units"]
                    row["case_input_id"] = reference.case_input_id
                    row["simulation_case_id"] = reference.simulation_case_id
                    row["input_admission"] = "metadata admitted; adapter bytes checked when selected"
                else:
                    row["input_admission"] = "no admitted input metadata at this checkpoint"
                rows.append(row)
    return CampaignOutcomeCatalog(tuple(rows), tuple(issues))


def query_campaign_outcomes(
    catalog: CampaignOutcomeCatalog,
    *,
    task: str,
    source_name: str,
    material_family: str | None = None,
    outcome: str | None = None,
    origin: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select exact candidate evidence without changing success membership."""
    if outcome not in {None, "pending", "successful", "failed"}:
        message = "Outcome filter must be pending, successful, failed, or None."
        raise ValueError(message)
    filters = {"task": task, "source_name": source_name, "material_family": material_family, "outcome": outcome, "origin": origin}
    return tuple(row for row in catalog.rows if all(value is None or row.get(key) == value for key, value in filters.items()))


def campaign_outcome_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Project attempts and scientific outcomes separately for human inspection, then every sampled parameter as its own column."""
    records = []
    for row in rows:
        failure = row.get("failure") or {}
        parameters = row.get("parameters")
        records.append(
            {
                **{key: row.get(key) for key in _CANDIDATE_COLUMNS},
                "attempts": len(row.get("attempts", ())),
                "failure_stage": failure.get("stage"),
                "failure_class": failure.get("class"),
                "reason_code": failure.get("reason_code"),
                "input_admission": row.get("input_admission"),
                **(dict(parameters) if isinstance(parameters, Mapping) else {}),
            }
        )
    return pd.DataFrame(records)


def campaign_outcome_parameters(table: pd.DataFrame) -> tuple[str, ...]:
    """Return the sampled-parameter columns of a campaign outcome table in table order."""
    return tuple(str(column) for column in table.columns if column not in CAMPAIGN_OUTCOME_COLUMNS)


def campaign_outcome_rates(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Use all terminal candidates as the rate denominator; keep pending visible."""
    grouped: dict[tuple[str, str, str, str], dict[str, int]] = {}
    keys = ("material_family", "domain_profile", "origin", "design_key")
    for row in rows:
        key = (str(row["material_family"]), str(row["domain_profile"]), str(row["origin"]), str(row["design_key"]))
        counts = grouped.setdefault(key, {"successful": 0, "failed": 0, "pending": 0})
        counts[str(row["outcome"])] += 1
    values = []
    for key, counts in sorted(grouped.items()):
        terminal = counts["successful"] + counts["failed"]
        values.append(
            dict(zip(keys, key, strict=True)) | counts | {"success_fraction_terminal": counts["successful"] / terminal if terminal else None}
        )
    return pd.DataFrame(values)


def campaign_failure_rates(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Count terminal reasons against all terminal candidates in the same group."""
    candidates = tuple(rows)
    keys = ("material_family", "domain_profile", "origin", "design_key")
    totals: dict[tuple[str, ...], int] = {}
    failures: dict[tuple[str, ...], int] = {}
    for row in candidates:
        group = tuple(str(row[key]) for key in keys)
        outcome = row["outcome"]
        if outcome in {"successful", "failed"}:
            totals[group] = totals.get(group, 0) + 1
        if outcome == "failed":
            failure = row.get("failure") or {}
            reason = tuple(str(failure.get(key) or "unrecorded") for key in ("stage", "class", "reason_code"))
            key = group + reason
            failures[key] = failures.get(key, 0) + 1
    records = []
    for key, count in sorted(failures.items()):
        group = key[: len(keys)]
        records.append(
            dict(zip((*keys, "failure_stage", "failure_class", "reason_code"), key, strict=True))
            | {"failed": count, "terminal_candidates": totals[group], "reason_fraction_terminal": count / totals[group]}
        )
    return pd.DataFrame(records)


def outcome_parameter_values(rows: Iterable[Mapping[str, Any]], name: str, unit: str) -> dict[str, np.ndarray]:
    """Return actual finite scalar parameters grouped by observed candidate outcome."""
    groups: dict[str, list[float]] = {"successful": [], "failed": [], "pending": []}
    for row in rows:
        parameters = row.get("parameters")
        units = row.get("parameter_units")
        if not isinstance(parameters, Mapping) or not isinstance(units, Mapping) or units.get(name) != unit:
            continue
        value = parameters.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
            groups[str(row["outcome"])].append(float(value))
    return {outcome: np.asarray(values, dtype=np.float64) for outcome, values in groups.items()}
