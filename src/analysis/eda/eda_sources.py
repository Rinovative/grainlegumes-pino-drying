"""
eda_sources.py

Discover read-only generated-output EDA sources from terminal batches and case-local campaigns.

Responsibilities:
  - Admit terminal batch manifests through Generation's strict terminal validator
  - Admit partial campaign cases independently while isolating invalid siblings
  - Preserve deterministic source, batch, and case accounting for EDA summaries
  - Deduplicate exact duplicate evidence while rejecting conflicting identities

Design principles:
  - Generation owns all manifest, campaign, and case publication validation
  - Terminal batch evidence remains authoritative whenever it is available
  - Partial case failures never suppress separately valid completed siblings

This module does NOT:
  - Finalize batches, publish Dataset packages, or change campaign state
  - Reimplement Generation schema, hash, or membership validation
  - Read training, checkpoint, experiment, or evaluation artifacts
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from src import common, generation

if TYPE_CHECKING:
    from src.generation.cases.generation_cases_config import GenerationConfig
    from src.generation.runtime.generation_runtime_batch import TerminalBatchEvidence, TerminalCaseEvidence

_MAXIMUM_DISCOVERY_ISSUES = 20
_MAXIMUM_ISSUE_MESSAGE_CHARACTERS = 512
_SourceKind = Literal["terminal", "partial"]


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDAIssue:
    """Describe one bounded source or case-local admission issue."""

    source_id: str
    message: str


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDABatch:
    """Hold one terminal or partial batch and its EDA-admitted completed cases."""

    source_kind: _SourceKind
    generation_root: Path
    batch_id: str
    batch_name: str
    batch_storage_name: str
    simulation_profile: str
    available_learning_views: tuple[str, ...]
    airflow_source: str
    material_family: str
    sampling_regime: str
    template_sha256: str
    git_commit: str
    scientific_values: dict[str, Any]
    cases: tuple[TerminalCaseEvidence, ...]
    failed_case_indices: tuple[int, ...]
    incomplete_case_indices: tuple[int, ...]
    invalid_case_indices: tuple[int, ...]
    campaign_sources: tuple[tuple[str, str], ...] = ()
    terminal_evidence: TerminalBatchEvidence | None = None
    config: GenerationConfig | None = None

    def __post_init__(self) -> None:
        """Require one complete, non-overlapping configured-case accounting."""
        valid_indices = tuple(case.case_index for case in self.cases)
        partitions = {
            "valid": valid_indices,
            "failed": self.failed_case_indices,
            "incomplete": self.incomplete_case_indices,
            "invalid": self.invalid_case_indices,
        }
        for label, indices in partitions.items():
            if tuple(sorted(set(indices))) != indices:
                message = f"Generated-output EDA {label} case indices must be unique and ordered."
                raise ValueError(message)
        all_indices = tuple(index for indices in partitions.values() for index in indices)
        if len(set(all_indices)) != len(all_indices):
            message = f"Generated-output EDA case outcomes overlap for batch {self.batch_id!r}."
            raise ValueError(message)
        if self.source_kind == "terminal":
            if self.terminal_evidence is None or self.config is not None or self.campaign_sources:
                message = "Terminal EDA evidence must retain only its authoritative terminal batch."
                raise ValueError(message)
        else:
            if self.terminal_evidence is not None or self.config is None or not self.campaign_sources:
                message = "Partial EDA evidence requires configuration and campaign-source identities."
                raise ValueError(message)
            expected = tuple(sorted(self.config.case_indices))
            if tuple(sorted(all_indices)) != expected:
                message = f"Partial EDA accounting does not cover configured batch {self.batch_id!r}."
                raise ValueError(message)
            run_ids = tuple(run_id for run_id, _state in self.campaign_sources)
            if tuple(sorted(self.campaign_sources)) != self.campaign_sources or len(set(run_ids)) != len(run_ids):
                message = f"Partial EDA campaign sources must be unique and ordered for batch {self.batch_id!r}."
                raise ValueError(message)

    def _scientific_role(self, name: str) -> str | None:
        """Return one persisted batch-role identifier without inference."""
        value = self.scientific_values.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            message = f"Generated-output EDA {name} evidence must be non-empty text."
            raise TypeError(message)
        return value

    @property
    def campaign_purpose(self) -> str | None:
        """Return the persisted campaign purpose when declared."""
        value = self._scientific_role("campaign_purpose")
        terminal = None if self.terminal_evidence is None else self.terminal_evidence.campaign_purpose
        if value is not None and terminal is not None and value != terminal:
            message = f"Generated-output EDA campaign purpose disagrees for {self.batch_id!r}."
            raise ValueError(message)
        return terminal if terminal is not None else value

    @property
    def material_role(self) -> str | None:
        """Return the authoritative batch material/source role when declared."""
        value = self._scientific_role("material_role")
        configured = None if self.config is None else self.config.material_role
        if value is not None and configured is not None and value != configured:
            message = f"Generated-output EDA material role disagrees for {self.batch_id!r}."
            raise ValueError(message)
        return configured if configured is not None else value

    @property
    def evaluation_regime(self) -> str | None:
        """Return the authoritative batch evaluation regime when declared."""
        value = self._scientific_role("evaluation_regime")
        configured = None if self.config is None else self.config.evaluation_regime
        if value is not None and configured is not None and value != configured:
            message = f"Generated-output EDA evaluation regime disagrees for {self.batch_id!r}."
            raise ValueError(message)
        return configured if configured is not None else value

    @property
    def available_case_count(self) -> int:
        """Return individually admitted completed cases visible to EDA."""
        return len(self.cases)

    @property
    def failed_case_count(self) -> int:
        """Return cases with authoritative failure evidence and no valid publication."""
        return len(self.failed_case_indices)

    @property
    def incomplete_case_count(self) -> int:
        """Return configured cases with no valid, failed, or corrupt terminal evidence."""
        return len(self.incomplete_case_indices)

    @property
    def invalid_case_count(self) -> int:
        """Return cases whose concrete candidate evidence failed validation."""
        return len(self.invalid_case_indices)

    @property
    def discovered_case_count(self) -> int:
        """Return every configured case represented by this source."""
        return self.available_case_count + self.failed_case_count + self.incomplete_case_count + self.invalid_case_count

    @property
    def campaign_run_id(self) -> str | None:
        """Return the campaign run only when this view has one source."""
        return self.campaign_sources[0][0] if len(self.campaign_sources) == 1 else None

    @property
    def campaign_state(self) -> str | None:
        """Return the campaign state only when this view has one source."""
        return self.campaign_sources[0][1] if len(self.campaign_sources) == 1 else None

    @property
    def interpreter_batch(self) -> TerminalBatchEvidence | GeneratedOutputEDABatch:
        """Return exact terminal evidence when present, otherwise this partial view."""
        return self.terminal_evidence if self.terminal_evidence is not None else self

    def case(self, case_id: str) -> TerminalCaseEvidence:
        """Return one uniquely admitted case by canonical identifier."""
        matches = tuple(case for case in self.cases if case.case_id == case_id)
        if len(matches) != 1:
            message = f"Generated-output EDA batch {self.batch_id!r} has no unique case {case_id!r}."
            raise ValueError(message)
        return matches[0]

    def scientific_config_payload(self) -> dict[str, Any]:
        """Return an independent resolved scientific configuration copy."""
        if self.terminal_evidence is not None:
            return self.terminal_evidence.scientific_config_payload()
        return copy.deepcopy(self.scientific_values)


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDACatalog:
    """Hold deterministic EDA-visible batches, discovery counts, and bounded issues."""

    batches: tuple[GeneratedOutputEDABatch, ...]
    issues: tuple[GeneratedOutputEDAIssue, ...]
    discovered_batch_count: int
    complete_batch_count: int
    partial_batch_count: int
    total_issue_count: int


def _issue(source_id: str, error: BaseException | str) -> GeneratedOutputEDAIssue:
    """Build one bounded diagnostic issue without changing its failure category."""
    message = str(error)
    if len(message) > _MAXIMUM_ISSUE_MESSAGE_CHARACTERS:
        message = f"{message[: _MAXIMUM_ISSUE_MESSAGE_CHARACTERS - 3]}..."
    return GeneratedOutputEDAIssue(source_id=source_id, message=message)


def _case_signature(case: TerminalCaseEvidence) -> tuple[object, ...]:
    """Return exact admitted case identity for duplicate comparison."""
    return (
        case.case_index,
        case.case_id,
        case.case_input_id,
        case.simulation_case_id,
        case.success_sha256,
        case.provenance_sha256,
        case.case_hdf5_sha256,
    )


def _batch_identity_signature(batch: GeneratedOutputEDABatch) -> tuple[object, ...]:
    """Return immutable batch identity shared by complete and partial views."""
    return (
        batch.batch_id,
        batch.batch_name,
        batch.batch_storage_name,
        batch.simulation_profile,
        batch.available_learning_views,
        batch.airflow_source,
        batch.material_family,
        batch.sampling_regime,
        batch.template_sha256,
        batch.git_commit,
        common.serialization.canonical_json_sha256(batch.scientific_values),
    )


def _batch_signature(batch: GeneratedOutputEDABatch) -> tuple[object, ...]:
    """Return evidence that must agree across exact duplicate source paths."""
    return (
        _batch_identity_signature(batch),
        tuple(_case_signature(case) for case in batch.cases),
        batch.failed_case_indices,
        batch.incomplete_case_indices,
        batch.invalid_case_indices,
    )


def _case_signatures_by_index(batch: GeneratedOutputEDABatch) -> dict[int, tuple[object, ...]]:
    """Index unique admitted case identities for cross-source subset checks."""
    signatures = {case.case_index: _case_signature(case) for case in batch.cases}
    if len(signatures) != len(batch.cases):
        message = f"Generated-output evidence repeats a case index in batch {batch.batch_id!r}."
        raise ValueError(message)
    return signatures


def _terminal_storage_names(storage_root: Path) -> tuple[str, ...]:
    """Return candidate terminal metadata directories without parsing manifests."""
    root = common.paths.get_generation_meta_root(storage_root=storage_root)
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        message = f"Generation metadata root is unsafe: {root}"
        raise ValueError(message)
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink() and (path / "batch_manifest.json").is_file()))


def _run_ids(storage_root: Path) -> tuple[str, ...]:
    """Return sorted persisted campaign-run directory locators."""
    root = common.paths.get_generation_meta_root(storage_root=storage_root) / "campaigns"
    if not root.exists():
        return ()
    if not root.is_dir() or root.is_symlink():
        message = f"Campaign metadata root is unsafe: {root}"
        raise ValueError(message)
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()))


def _terminal_batch(evidence: TerminalBatchEvidence) -> GeneratedOutputEDABatch:
    """Adapt exact terminal evidence without replacing its authoritative object."""
    return GeneratedOutputEDABatch(
        source_kind="terminal",
        generation_root=evidence.generation_root,
        batch_id=evidence.batch_id,
        batch_name=evidence.batch_name,
        batch_storage_name=evidence.batch_storage_name,
        simulation_profile=evidence.simulation_profile,
        available_learning_views=evidence.available_learning_views,
        airflow_source=evidence.airflow_source,
        material_family=evidence.material_family,
        sampling_regime=evidence.sampling_regime,
        template_sha256=evidence.template_sha256,
        git_commit=evidence.git_commit,
        scientific_values=evidence.scientific_config_payload(),
        cases=evidence.cases,
        failed_case_indices=(),
        incomplete_case_indices=(),
        invalid_case_indices=(),
        terminal_evidence=evidence,
    )


def _partial_batch(
    config: GenerationConfig,
    *,
    storage_root: Path,
    run_id: str,
    campaign_state: str,
    campaign_git_commit: str,
    successful_indices: tuple[int, ...],
    failed_case_indices: tuple[int, ...],
    incomplete_case_indices: tuple[int, ...],
    initial_invalid_case_indices: tuple[int, ...] = (),
) -> tuple[GeneratedOutputEDABatch, tuple[GeneratedOutputEDAIssue, ...]]:
    """Individually admit declared completed cases while isolating invalid siblings."""
    admitted: list[TerminalCaseEvidence] = []
    issues: list[GeneratedOutputEDAIssue] = []
    invalid = list(initial_invalid_case_indices)
    for case_index in successful_indices:
        case_id = config.case_id(case_index)
        try:
            admitted.append(
                generation.runtime.admit_completed_case(
                    config,
                    case_index,
                    storage_root=storage_root,
                    validation_depth="full",
                    git_commit=campaign_git_commit,
                )
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            invalid.append(case_index)
            issues.append(_issue(f"{run_id}:{case_id}", error))
    return (
        GeneratedOutputEDABatch(
            source_kind="partial",
            generation_root=common.paths.get_generation_root(storage_root=storage_root).resolve(),
            batch_id=config.batch_id,
            batch_name=config.batch_name,
            batch_storage_name=config.batch_storage_name,
            simulation_profile=config.profile.id,
            available_learning_views=config.profile.available_learning_views,
            airflow_source=config.profile.airflow_source,
            material_family=config.material_family,
            sampling_regime=config.sampling_regime,
            template_sha256=config.template_sha256,
            git_commit=campaign_git_commit,
            scientific_values=copy.deepcopy(config.scientific_values),
            cases=tuple(admitted),
            failed_case_indices=tuple(sorted(failed_case_indices)),
            incomplete_case_indices=tuple(sorted(incomplete_case_indices)),
            invalid_case_indices=tuple(sorted(invalid)),
            campaign_sources=((run_id, campaign_state),),
            config=config,
        ),
        tuple(issues),
    )


def _unfinished_case_classification(
    config: GenerationConfig,
    *,
    storage_root: Path,
    run_id: str,
    campaign_git_commit: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[GeneratedOutputEDAIssue, ...]]:
    """Classify a non-final campaign through case-local Generation validators."""
    successful: list[int] = []
    failed: list[int] = []
    incomplete: list[int] = []
    invalid: list[int] = []
    issues: list[GeneratedOutputEDAIssue] = []
    for case_index in config.case_indices:
        case_id = config.case_id(case_index)
        try:
            valid = generation.runtime.completed_case_is_valid(
                config,
                case_index,
                storage_root=storage_root,
                git_commit=campaign_git_commit,
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            invalid.append(case_index)
            issues.append(_issue(f"{run_id}:{case_id}", error))
            continue
        if valid:
            successful.append(case_index)
            continue
        try:
            recorded_failure = generation.runtime.case_failure_is_recorded(
                config,
                case_index,
                storage_root=storage_root,
                execution_run_id=run_id,
                git_commit=campaign_git_commit,
            )
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
            invalid.append(case_index)
            issues.append(_issue(f"{run_id}:{case_id}", error))
            continue
        if recorded_failure:
            failed.append(case_index)
        else:
            incomplete.append(case_index)
    return tuple(successful), tuple(failed), tuple(incomplete), tuple(invalid), tuple(issues)


def _admit_campaign(
    run_id: str,
    *,
    storage_root: Path,
) -> tuple[tuple[GeneratedOutputEDABatch, ...], tuple[GeneratedOutputEDAIssue, ...]]:
    """Discover one campaign's partial case sources without task filtering."""
    manifest = generation.publication.campaign_evidence.load_campaign_run(run_id, storage_root=storage_root)
    campaign = generation.publication.campaign_evidence.campaign_from_manifest(
        manifest,
        require_executable=False,
    )
    partial = generation.campaign.read_partial_campaign_diagnostic_evidence(run_id, storage_root=storage_root)
    records_by_batch: dict[str, tuple[dict[str, Any], ...]] = {}
    campaign_state = str(manifest["state"])
    if partial is not None:
        all_records = [*partial["successful_cases"], *partial["failed_cases"]]
        records_by_batch = {
            name: tuple(
                sorted(
                    (record for record in all_records if record["batch_name"] == name),
                    key=lambda item: int(item["case_index"]),
                )
            )
            for name in {str(record["batch_name"]) for record in all_records}
        }
        campaign_state = str(partial["campaign_state"])
    batches: list[GeneratedOutputEDABatch] = []
    issues: list[GeneratedOutputEDAIssue] = []
    for config in campaign.batches:
        initial_invalid: tuple[int, ...] = ()
        if partial is not None:
            batch_records = records_by_batch.get(config.batch_name)
            if batch_records is None:
                message = f"Partial campaign evidence omits configured batch {config.batch_name!r}."
                raise RuntimeError(message)
            successful = tuple(int(record["case_index"]) for record in batch_records if record["state"] == "successful")
            failed = tuple(int(record["case_index"]) for record in batch_records if record["state"] == "failed")
            recorded_indices = {int(record["case_index"]) for record in batch_records}
            incomplete = tuple(index for index in config.case_indices if index not in recorded_indices)
        else:
            successful, failed, incomplete, initial_invalid, classification_issues = _unfinished_case_classification(
                config,
                storage_root=storage_root,
                run_id=run_id,
                campaign_git_commit=str(manifest["git_commit"]),
            )
            issues.extend(classification_issues)
        batch, batch_issues = _partial_batch(
            config,
            storage_root=storage_root,
            run_id=run_id,
            campaign_state=campaign_state,
            campaign_git_commit=str(manifest["git_commit"]),
            successful_indices=successful,
            failed_case_indices=failed,
            incomplete_case_indices=incomplete,
            initial_invalid_case_indices=initial_invalid,
        )
        batches.append(batch)
        issues.extend(batch_issues)
    return tuple(batches), tuple(issues)


def _configured_case_indices(batch: GeneratedOutputEDABatch) -> tuple[int, ...]:
    """Return the complete configured membership behind one admitted batch view."""
    if batch.config is not None:
        return tuple(sorted(batch.config.case_indices))
    return tuple(case.case_index for case in batch.cases)


def _merge_batch(
    batches: dict[str, GeneratedOutputEDABatch],
    candidate: GeneratedOutputEDABatch,
) -> GeneratedOutputEDAIssue | None:
    """Merge compatible case evidence without double-counting configured outcomes."""
    current = batches.get(candidate.batch_id)
    if current is None:
        batches[candidate.batch_id] = candidate
        return None
    conflict = "Conflicting generated-output evidence was discovered for one batch identity."
    if _batch_identity_signature(current) != _batch_identity_signature(candidate) or _configured_case_indices(current) != _configured_case_indices(
        candidate
    ):
        return _issue(candidate.batch_id, conflict)
    current_cases = _case_signatures_by_index(current)
    candidate_cases = _case_signatures_by_index(candidate)
    overlap = set(current_cases).intersection(candidate_cases)
    if any(current_cases[index] != candidate_cases[index] for index in overlap):
        return _issue(candidate.batch_id, conflict)
    if current.source_kind == "terminal" or candidate.source_kind == "terminal":
        terminal = current if current.source_kind == "terminal" else candidate
        partial = candidate if current.source_kind == "terminal" else current
        terminal_indices = set(_case_signatures_by_index(terminal))
        if partial.source_kind == "partial" and set(_case_signatures_by_index(partial)).issubset(terminal_indices):
            batches[candidate.batch_id] = terminal
            return None
        if current.source_kind == candidate.source_kind == "terminal" and _batch_signature(current) == _batch_signature(candidate):
            return None
        return _issue(candidate.batch_id, conflict)

    campaign_states: dict[str, str] = {}
    for run_id, state in (*current.campaign_sources, *candidate.campaign_sources):
        previous = campaign_states.setdefault(run_id, state)
        if previous != state:
            return _issue(candidate.batch_id, conflict)
    cases_by_index = {case.case_index: case for case in (*current.cases, *candidate.cases)}
    valid_indices = set(cases_by_index)
    remaining = set(_configured_case_indices(current)).difference(valid_indices)
    invalid_evidence = set(current.invalid_case_indices).union(candidate.invalid_case_indices)
    invalid_indices = remaining.intersection(invalid_evidence)
    remaining.difference_update(invalid_indices)
    failed_evidence = set(current.failed_case_indices).union(candidate.failed_case_indices)
    failed_indices = remaining.intersection(failed_evidence)
    remaining.difference_update(failed_indices)
    batches[candidate.batch_id] = replace(
        current,
        cases=tuple(cases_by_index[index] for index in sorted(cases_by_index)),
        failed_case_indices=tuple(sorted(failed_indices)),
        incomplete_case_indices=tuple(sorted(remaining)),
        invalid_case_indices=tuple(sorted(invalid_indices)),
        campaign_sources=tuple(sorted(campaign_states.items())),
    )
    return None


def discover_generated_output_eda_catalog(
    *,
    storage_root: Path | str,
) -> GeneratedOutputEDACatalog:
    """
    Discover terminal and partial generated outputs for read-only unified EDA.

    Parameters
    ----------
    storage_root : Path | str
        Canonical storage root containing Generation metadata and publications.

    Returns
    -------
    GeneratedOutputEDACatalog
        Deterministic unique batches, complete/partial counts, and bounded issues.

    """
    root = Path(storage_root).expanduser().resolve()
    batches: dict[str, GeneratedOutputEDABatch] = {}
    issues: list[GeneratedOutputEDAIssue] = []
    discovered_batch_ids: set[str] = set()
    for storage_name in _terminal_storage_names(root):
        try:
            evidence = generation.runtime.admit_terminal_batch(storage_name, storage_root=root, validation_depth="routine")
            discovered_batch_ids.add(evidence.batch_id)
            issue = _merge_batch(batches, _terminal_batch(evidence))
            if issue is not None:
                issues.append(issue)
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:  # noqa: PERF203 -- source isolation is intentional.
            issues.append(_issue(storage_name, error))
    for run_id in _run_ids(root):
        try:
            candidates, campaign_issues = _admit_campaign(run_id, storage_root=root)
            discovered_batch_ids.update(candidate.batch_id for candidate in candidates)
            issues.extend(campaign_issues)
            for candidate in candidates:
                issue = _merge_batch(batches, candidate)
                if issue is not None:
                    issues.append(issue)
        except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:  # noqa: PERF203 -- campaign isolation is intentional.
            issues.append(_issue(run_id, error))
    ordered = tuple(sorted(batches.values(), key=lambda batch: (batch.material_family, batch.sampling_regime, batch.batch_id)))
    return GeneratedOutputEDACatalog(
        batches=ordered,
        issues=tuple(issues[:_MAXIMUM_DISCOVERY_ISSUES]),
        discovered_batch_count=len(discovered_batch_ids),
        complete_batch_count=sum(batch.source_kind == "terminal" for batch in ordered),
        partial_batch_count=sum(batch.source_kind == "partial" and batch.available_case_count > 0 for batch in ordered),
        total_issue_count=len(issues),
    )
