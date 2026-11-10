"""Resolve which Datasets and cases each artifact role of an evaluable run covers, the same way for both tasks.

Responsibilities:
  - Project the admitted split of a completed run into its ID-test role and its non-empty material-OOD and
    parameter-OOD roles
  - Keep every OOD Evaluation source in split order with its identity, ordered cases, membership digest and assignment
  - Name each role's artifact Dataset and folder and project its shared provenance block

Design principles:
  - The persisted split, admitted by the one split loader, is the only membership authority
  - Both tasks share one role plan; task readers only open and check the parts it names
  - An empty OOD role is not planned, so it gets no folder, artifact or metric
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

from src import common, datasets

from . import analysis_artifact_contracts as contracts

ArtifactSplit = Literal["eval", "material_ood", "parameter_ood"]
ArtifactDatasetRole = Literal["id", "material_ood", "parameter_ood"]
# Artifact splits in canonical order; an OOD split names its Dataset role and saved split role alike.
ARTIFACT_SPLITS: tuple[ArtifactSplit, ...] = ("eval", "material_ood", "parameter_ood")
ARTIFACT_DATASET_ROLES: tuple[ArtifactDatasetRole, ...] = ("id", "material_ood", "parameter_ood")


def dataset_role_of(split: ArtifactSplit) -> ArtifactDatasetRole:
    """Return the Dataset role of one artifact split: ``id`` for ``eval``, the OOD role itself otherwise."""
    return "id" if split == "eval" else split


def split_of(dataset_role: ArtifactDatasetRole) -> ArtifactSplit:
    """Return the artifact split of one Dataset role: ``eval`` for ``id``, the OOD role itself otherwise."""
    return "eval" if dataset_role == "id" else dataset_role


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
        return "id_test" if self.split == "eval" else self.split

    def root(self, run_dir: Path | str) -> Path:
        """Return this role's artifact folder: ``analysis/id`` or ``analysis/<OOD role>/<dataset name>``."""
        if self.split == "eval":
            return common.paths.resolve_id_analysis_dir(run_dir)
        return common.paths.resolve_ood_analysis_dir(run_dir, self.split, self.dataset_name)

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
    """Bind one completed run of either task to its ID-test role and its non-empty OOD Evaluation roles."""

    run_dir: Path
    run_name: str
    task: str
    id_role: ArtifactRolePlan
    ood_roles: Mapping[ArtifactSplit, ArtifactRolePlan]

    @property
    def roles(self) -> tuple[ArtifactRolePlan, ...]:
        """Return the ID-test role and every planned OOD role in canonical order."""
        return (self.id_role, *self.ood_roles.values())

    @property
    def dataset_roles(self) -> tuple[ArtifactDatasetRole, ...]:
        """Return the Dataset roles this run has artifacts for, in canonical order."""
        return tuple(role.dataset_role for role in self.roles)

    def role(self, split: ArtifactSplit) -> ArtifactRolePlan:
        """Return the role of one artifact split, refusing an OOD role the run does not have."""
        if split == "eval":
            return self.id_role
        if split not in ARTIFACT_SPLITS:
            msg = f"Unknown artifact split {split!r}."
            raise ValueError(msg)
        if split not in self.ood_roles:
            msg = f"Run {self.run_name!r} has no saved {split} artifact role."
            raise FileNotFoundError(msg)
        return self.ood_roles[split]


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
        ID-test role over the training Dataset and one role per non-empty OOD Evaluation role (``material_ood``,
        ``parameter_ood``) over its sources in split order.

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
    ood_roles: dict[ArtifactSplit, ArtifactRolePlan] = {
        role: _role_plan(
            split=role,
            dataset_role=role,
            dataset_name=str(split.ood_name(role)),
            parts=split.ood_parts(role),
            assignments=split.evaluation_assignments[role],
        )
        for role in split.ood_roles
    }
    return ArtifactPlan(
        run_dir=Path(run_dir).expanduser().resolve(),
        run_name=str(run["name"]),
        task=split.task,
        id_role=id_role,
        ood_roles=ood_roles,
    )
