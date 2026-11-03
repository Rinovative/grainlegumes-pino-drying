"""Discover role-neutral Generation campaigns and published Dataset selections.

Responsibilities:
  - Discover successful campaign cases through Generation catalog admission
  - Group exact cases by campaign, material and domain for EDA selection
  - Collect admitted Dataset manifest metadata and bounded candidate issues

Design principles:
  - Generation owns source outcomes and exact source admission
  - Discovery keeps metadata light until a selected scientific case is opened
  - Unrelated discovery candidates remain inspectable after bounded failures

This module does NOT:
  - Open source numerical arrays or generate campaign cases
  - Publish Dataset selections or assign experiment Train and Evaluation roles
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import common, datasets, generation

_MAXIMUM_DISCOVERY_ISSUES = 20
_MAXIMUM_ISSUE_MESSAGE_CHARACTERS = 512
_TASKS = ("steady_flow", "transient_drying")
_SOURCE_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDAIssue:
    """Describe one bounded campaign or package admission issue."""

    source_id: str
    message: str


@dataclass(frozen=True, slots=True)
class PublishedDatasetEDACatalog:
    """Retain admitted immutable package metadata before scientific payload reads."""

    packages: tuple[dict[str, Any], ...]
    issues: tuple[GeneratedOutputEDAIssue, ...]


def discover_published_dataset_eda_catalog(*, storage_root: Path) -> PublishedDatasetEDACatalog:
    """Discover package metadata through its owner without opening source arrays."""
    root = common.paths.get_datasets_root(storage_root=storage_root)
    packages: list[dict[str, Any]] = []
    issues: list[GeneratedOutputEDAIssue] = []
    if not root.exists():
        return PublishedDatasetEDACatalog((), ())
    if root.is_symlink() or not root.is_dir():
        message = f"Dataset metadata root is unsafe: {root}."
        raise ValueError(message)
    for task_directory in sorted(root.iterdir()):
        if task_directory.name.startswith(".") or task_directory.is_symlink() or not task_directory.is_dir():
            continue
        for directory in sorted(task_directory.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                continue
            try:
                manifest = datasets.packages.manifest.load_package_manifest_evidence(task_directory.name, directory.name, storage_root=storage_root)
            except (OSError, ValueError, TypeError) as error:
                issues.append(GeneratedOutputEDAIssue(f"{task_directory.name}/{directory.name}", str(error)[:_MAXIMUM_ISSUE_MESSAGE_CHARACTERS]))
                continue
            packages.append(manifest)
    return PublishedDatasetEDACatalog(tuple(packages), tuple(issues))


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDACase:
    """Retain one immutable candidate row and its view-local case number."""

    case_id: str
    case_index: int
    row: dict[str, Any]

    @property
    def selector(self) -> dict[str, Any]:
        """Return the seven exact identities required for Source admission."""
        return {key: self.row[key] for key in _SOURCE_SELECTOR_KEYS}


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDABatch:
    """Expose one campaign/material/domain selection without assigning a role."""

    generation_root: Path
    batch_id: str
    campaign_name: str
    source_name: str
    campaign_digest: str
    campaign_purpose: str
    simulation_profile: str
    available_learning_views: tuple[str, ...]
    material_family: str
    domain_profile: str
    sampling_regime: str
    cases: tuple[GeneratedOutputEDACase, ...]
    candidate_rows: tuple[dict[str, Any], ...]
    storage_root: Path

    def __post_init__(self) -> None:
        """Require an exact non-overlapping candidate inventory."""
        if not self.candidate_rows or len({case.case_id for case in self.cases}) != len(self.cases):
            message = "Generation EDA campaign view requires candidate decisions and unique successful cases."
            raise ValueError(message)
        if tuple(case.case_index for case in self.cases) != tuple(range(1, len(self.cases) + 1)):
            message = "Generation EDA case positions must be contiguous and view-local."
            raise ValueError(message)
        if any(row["campaign_digest"] != self.campaign_digest or row["source_name"] != self.source_name for row in self.candidate_rows):
            message = "Generation EDA candidates disagree with campaign identity."
            raise ValueError(message)

    @property
    def available_case_count(self) -> int:
        """Return successful cases available for lazy exact admission."""
        return len(self.cases)

    @property
    def selected_case_count(self) -> int:
        """Return successful cases selected by a sealed campaign pool."""
        return sum(row["outcome"] == "successful" and row["selected"] for row in self.candidate_rows)

    @property
    def failed_case_count(self) -> int:
        """Return authored failed candidate decisions."""
        return sum(row["outcome"] == "failed" for row in self.candidate_rows)

    @property
    def incomplete_case_count(self) -> int:
        """Return candidates without a final decision."""
        return sum(row["outcome"] == "pending" for row in self.candidate_rows)

    @property
    def invalid_case_count(self) -> int:
        """Return invalid rows; catalog validation raises before this view exists."""
        return 0

    @property
    def discovered_case_count(self) -> int:
        """Return all primary, supplemental, and replacement candidates."""
        return len(self.candidate_rows)

    def case(self, case_id: str) -> GeneratedOutputEDACase:
        """Return one exact source candidate by its authored case ID."""
        matches = tuple(case for case in self.cases if case.case_id == case_id)
        if len(matches) != 1:
            message = f"Generation EDA view {self.batch_id!r} has no unique case {case_id!r}."
            raise ValueError(message)
        return matches[0]

    def admit_case(self, case: GeneratedOutputEDACase) -> Any:
        """Admit the selected successful source only when its payload is opened."""
        if case not in self.cases:
            message = "Generation EDA case does not belong to this campaign view."
            raise ValueError(message)
        return generation.publication.catalog.resolve_raw_case_reference(
            self.simulation_profile,
            case.selector,
            storage_root=self.storage_root,
        )


@dataclass(frozen=True, slots=True)
class GeneratedOutputEDACatalog:
    """Hold deterministic campaign views and bounded discovery issues."""

    batches: tuple[GeneratedOutputEDABatch, ...]
    issues: tuple[GeneratedOutputEDAIssue, ...]
    discovered_campaign_count: int
    discovered_candidate_count: int
    total_issue_count: int


def _issue(source_id: str, error: BaseException | str) -> GeneratedOutputEDAIssue:
    """Build one bounded source issue without hiding its cause."""
    message = str(error)
    if len(message) > _MAXIMUM_ISSUE_MESSAGE_CHARACTERS:
        message = f"{message[: _MAXIMUM_ISSUE_MESSAGE_CHARACTERS - 3]}..."
    return GeneratedOutputEDAIssue(source_id=source_id, message=message)


def discover_generated_output_eda_catalog(*, storage_root: Path) -> GeneratedOutputEDACatalog:
    """List current campaigns and successful candidates without opening arrays."""
    root = Path(storage_root).expanduser().resolve()
    views: list[GeneratedOutputEDABatch] = []
    issues: list[GeneratedOutputEDAIssue] = []
    campaign_count = 0
    candidate_count = 0
    for task_id in _TASKS:
        try:
            campaigns = generation.publication.catalog.list_campaigns(task_id, storage_root=root)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            issues.append(_issue(task_id, error))
            continue
        campaign_count += len(campaigns)
        for campaign in campaigns:
            campaign_name = str(campaign["campaign_name"])
            source_name = str(campaign["source_name"])
            try:
                rows = generation.publication.catalog.list_campaign_cases(task_id, source_name, storage_root=root)
                candidate_count += len(rows)
                by_material_domain: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
                for row in rows:
                    group_key = (str(row["material_family"]), str(row["domain_profile"]), str(row["sampling_regime"]))
                    by_material_domain.setdefault(group_key, []).append(dict(row))
                for (material, domain_profile, sampling_regime), candidates in sorted(by_material_domain.items()):
                    successful = tuple(row for row in candidates if row["outcome"] == "successful")
                    cases = tuple(GeneratedOutputEDACase(str(row["case_id"]), position, row) for position, row in enumerate(successful, start=1))
                    view_id = f"{task_id}__{source_name}__{material}__{domain_profile}__{sampling_regime}"
                    views.append(
                        GeneratedOutputEDABatch(
                            generation_root=common.paths.get_generation_root(storage_root=root),
                            batch_id=view_id,
                            campaign_name=campaign_name,
                            source_name=source_name,
                            campaign_digest=str(campaign["campaign_digest"]),
                            campaign_purpose=str(campaign["campaign_purpose"]),
                            simulation_profile=task_id,
                            available_learning_views=(task_id,),
                            material_family=material,
                            domain_profile=domain_profile,
                            sampling_regime=sampling_regime,
                            cases=cases,
                            candidate_rows=tuple(candidates),
                            storage_root=root,
                        )
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                issues.append(_issue(f"{task_id}:{campaign_name}", error))
    ordered = tuple(sorted(views, key=lambda view: (view.simulation_profile, view.campaign_name, view.material_family, view.domain_profile)))
    return GeneratedOutputEDACatalog(
        batches=ordered,
        issues=tuple(issues[:_MAXIMUM_DISCOVERY_ISSUES]),
        discovered_campaign_count=campaign_count,
        discovered_candidate_count=candidate_count,
        total_issue_count=len(issues),
    )
