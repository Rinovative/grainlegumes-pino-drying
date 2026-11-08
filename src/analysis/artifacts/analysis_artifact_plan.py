"""Resolve which Datasets and cases each artifact role of an evaluable run covers, the same way for both tasks.

Responsibilities:
  - Project the admitted split of a completed run into its ID-test role and its one additional-evaluation role
  - Keep every additional Evaluation source in split order with its identity, ordered cases, membership digest and assignment
  - Name each role's artifact Dataset and project its shared provenance block

Design principles:
  - The persisted split, admitted by the one split loader, is the only membership authority
  - Both tasks share one role plan; task readers only open and check the parts it names
  - Optuna trials are refused before any role is planned

This module does NOT:
  - Open Dataset payloads, run inference or write artifacts
  - Admit run bundles; callers pass already admitted completed-run evidence
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src import datasets

from . import analysis_artifact_contracts as contracts

ArtifactSplit = Literal["eval", "additional_evaluation"]
ArtifactDatasetRole = Literal["id", "additional_evaluation"]


@dataclass(frozen=True, slots=True)
class ArtifactRolePlan:
    """Bind one artifact role to its ordered source Datasets, their identities and their saved case membership."""

    split: ArtifactSplit
    dataset_role: ArtifactDatasetRole
    dataset_name: str
    source_dataset_ids: tuple[str, ...]
    source_identities: tuple[Mapping[str, Any], ...]
    case_ids_by_source: tuple[tuple[str, ...], ...]
    membership_digests: tuple[str, ...]
    evaluation_assignments: tuple[Mapping[str, Any], ...] = ()

    @property
    def split_role(self) -> datasets.preprocessing.splits.SplitRole:
        """Return the saved split role this artifact role reopens."""
        return "id_test" if self.split == "eval" else "additional_evaluation"

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Return member case IDs in split order across all sources."""
        return tuple(case_id for case_ids in self.case_ids_by_source for case_id in case_ids)

    @property
    def case_source_dataset_ids(self) -> tuple[str, ...]:
        """Return the source Dataset ID of every member case in split order."""
        return tuple(dataset_id for dataset_id, case_ids in zip(self.source_dataset_ids, self.case_ids_by_source, strict=True) for _ in case_ids)

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return member positions in the ordered concatenation of the sources' complete Datasets."""
        positions: list[int] = []
        offset = 0
        for identity, case_ids in zip(self.source_identities, self.case_ids_by_source, strict=True):
            position_by_id = {case_id: index for index, case_id in enumerate(identity["case_ids"])}
            positions.extend(offset + position_by_id[case_id] for case_id in case_ids)
            offset += len(identity["case_ids"])
        return tuple(positions)

    def dataset_provenance(self) -> dict[str, Any]:
        """Return the artifact provenance ``dataset`` block both tasks persist for this role."""
        return {
            "name": self.dataset_name,
            "role": self.dataset_role,
            "source_dataset_ids": list(self.source_dataset_ids),
            "source_identities": [dict(value) for value in self.source_identities],
            "case_ids_by_source": [list(values) for values in self.case_ids_by_source],
            "membership_digests": list(self.membership_digests),
            "evaluation_assignments": [dict(value) for value in self.evaluation_assignments],
        }


@dataclass(frozen=True, slots=True)
class ArtifactPlan:
    """Bind one completed run of either task to its ID-test role and optional additional-evaluation role."""

    run_dir: Path
    run_name: str
    task: str
    id_role: ArtifactRolePlan
    additional_role: ArtifactRolePlan | None

    def role(self, split: ArtifactSplit) -> ArtifactRolePlan:
        """Return the role of one artifact split, refusing an additional role the run does not have."""
        if split == "eval":
            return self.id_role
        if split != "additional_evaluation":
            msg = f"Unknown artifact split {split!r}."
            raise ValueError(msg)
        if self.additional_role is None:
            msg = f"Run {self.run_name!r} has no saved additional Evaluation artifact role."
            raise FileNotFoundError(msg)
        return self.additional_role


def _role_plan(
    *,
    split: ArtifactSplit,
    dataset_role: ArtifactDatasetRole,
    dataset_name: str,
    parts: Sequence[datasets.preprocessing.splits.SplitRoleEvidence],
    assignments: Sequence[Mapping[str, Any]],
) -> ArtifactRolePlan:
    """Build one role plan from admitted split parts in their persisted order."""
    case_ids = [case_id for part in parts for case_id in part.case_ids]
    if len(case_ids) != len(set(case_ids)):
        msg = f"Saved {dataset_role} sources share package case IDs; artifact case payloads would collide."
        raise ValueError(msg)
    dataset_ids = tuple(part.dataset_id for part in parts)
    return ArtifactRolePlan(
        split=split,
        dataset_role=dataset_role,
        dataset_name=dataset_name,
        source_dataset_ids=dataset_ids,
        source_identities=tuple(dict(part.dataset) for part in parts),
        case_ids_by_source=tuple(part.case_ids for part in parts),
        membership_digests=tuple(part.membership_digest for part in parts),
        evaluation_assignments=tuple(dict(item) for item in assignments),
    )


def artifact_plan_from_completed(completed: Mapping[str, Any], *, run_dir: Path | str) -> ArtifactPlan:
    """
    Project admitted completed-run evidence of either task into its artifact roles.

    Parameters
    ----------
    completed : Mapping[str, Any]
        Evidence returned by the evaluable- or completed-run admission (``config`` and ``split_indices``).
    run_dir : pathlib.Path | str
        Run directory the plan binds.

    Returns
    -------
    ArtifactPlan
        ID-test role over the training Dataset and, when the split has additional Evaluation sources, one
        additional-evaluation role over all of them in split order.

    Raises
    ------
    TypeError, ValueError
        If the run is an Optuna trial or its evidence is not shaped like an admitted run; config-to-split agreement
        belongs to that admission.

    """
    config = completed.get("config")
    raw_split = completed.get("split_indices")
    if not isinstance(config, Mapping) or not isinstance(raw_split, Mapping):
        msg = "Artifact planning requires admitted completed-run config and split evidence."
        raise TypeError(msg)
    run = config.get("run")
    if not isinstance(run, Mapping) or not isinstance(run.get("name"), str):
        msg = "Completed-run config lacks its scientific run name."
        raise TypeError(msg)
    contracts.refuse_optuna_trial(config, label=f"Run {run['name']}")
    split = datasets.preprocessing.splits.admit_split(raw_split)
    id_role = _role_plan(split="eval", dataset_role="id", dataset_name=split.id_test.dataset_id, parts=(split.id_test,), assignments=())
    additional_role = (
        None
        if split.additional_evaluation_name is None
        else _role_plan(
            split="additional_evaluation",
            dataset_role="additional_evaluation",
            dataset_name=split.additional_evaluation_name,
            parts=split.additional_evaluation,
            assignments=split.evaluation_assignments,
        )
    )
    return ArtifactPlan(
        run_dir=Path(run_dir).expanduser().resolve(),
        run_name=str(run["name"]),
        task=split.task,
        id_role=id_role,
        additional_role=additional_role,
    )
