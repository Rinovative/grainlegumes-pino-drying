"""Protect task-aware transient workflow dispatch and A0/A1/B lineage admission.

Responsibilities:
  - Check transient dispatch, exact role membership, wrapper closure, and A0/A1/B lineage

Design principles:
  - Comparisons require paired scientific identities and immutable B budget evidence

This module does NOT:
  - Train comparison arms or invent missing evaluation roles
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import analysis, common
from src.experiments import experiments_matched_budget as matched_budget

if TYPE_CHECKING:
    from src.analysis.artifacts.analysis_artifact_service import PreparedRunArtifacts
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession


def _split(arm: str) -> dict[str, Any]:
    """Represent complete saved cases with distinct strategy-owned runtime sampling."""
    roles = {}
    for role, cases in {
        "train": ["train_case"],
        "validation": ["validation_case"],
        "id_test": ["case_0001"],
        "scaling_train_one_step": ["train_case"],
    }.items():
        items = [f"{case}__{'window' if arm == 'b' and role != 'scaling_train_one_step' else 'step'}" for case in cases]
        roles[role] = {
            "case_ids": cases,
            "item_ids": items,
            "membership_digest": common.serialization.canonical_json_sha256({"case_ids": cases, "item_ids": items}),
        }
    roles["additional_evaluation"] = {"parts": []}
    return {
        "dataset_identity": {
            "train": {"dataset_id": "transient_dataset", "index_digest": "split", "data_contract_digest": "contract"},
            "additional_evaluation": [],
        },
        "roles": roles,
    }


def _matched_receipt() -> dict[str, Any]:
    """Build one matched train_seconds B receipt for test-owned comparison metadata."""
    return matched_budget.build_b_receipt(
        parent_identity_sha256="a" * 64,
        a0_handoff_sha256="c" * 64,
        b_config={
            "task": "transient_drying",
            "run": {"name": "run_b"},
            "training": {
                "teacher_handoff": {"source_run_name": "run_a0"},
                "matched_a1_b": {
                    "role": "b",
                    "train_seconds": 10.0,
                    "receipt_source_run_name": None,
                    "initialization": "a0_best_full_common_fresh_controller",
                },
            },
            "data": {
                "train_dataset": "transient_dataset",
                "additional_evaluation_datasets": [],
                "evaluation_assignments": [],
                "case_partition": {"kind": "exact", "train": ["train"], "validation": ["val"], "id_test": ["test"]},
                "spatial_stride": 1,
            },
            "input_profile": "canonical_physics_complete_v1",
            "temporal": {"temporal_conditioning": "time"},
            "model": {"kind": "fno", "params": {"hidden_channels": 8}},
            "evaluation": {"objective": {"id": "objective"}},
        },
        b_summary={
            "status": "completed",
            "terminal_controller": {
                "budget_complete": True,
                "clock_kind": "train_seconds",
                "train_seconds": 10.0,
                "successful_optimizer_steps": 10,
                "post_handoff_optimizer_steps": 10,
            },
            "terminal_epoch": 3,
            "best_epoch": 3,
            "best_checkpoint_sha256": "b" * 64,
            "last_checkpoint_sha256": "d" * 64,
        },
    )


def _frame(arm: str, *, parent: str = "parent") -> pd.DataFrame:
    matched = arm != "a0"
    controller: dict[str, Any] = {
        "arm": {"a0": "A0", "a1": "A1", "b": "B"}[arm],
        "stage": "stage_b_self_fed" if arm == "b" else "stage_a_teacher_forcing",
        "budget_control": "train_seconds" if matched else "stage_epochs",
        "clock_kind": "train_seconds" if matched else "optimizer_steps",
        "planned_stage_epochs": None if matched else 3,
        "completed_stage_epochs": 0 if matched else 3,
        "planned_train_seconds": 10.0 if matched else None,
        "train_seconds": 10.0 if matched else 0.0,
        "successful_optimizer_steps": 10 if matched else 3,
        "budget_complete": True,
        "best_within_budget_epoch": 2,
    }
    identity = {
        "model_kind": "fno",
        "model_parameters": {"hidden_channels": 8},
        "input_profile": "canonical_physics_complete_v1",
        "boundary_representation": "both_interval_endpoints_with_startup_support",
        "scaling_identity": {"semantic_digest": "scale"},
    }
    split = _split(arm)
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "transient_sequence_records": (SimpleNamespace(case_id="case_0001", identity=identity),),
            "transient_unavailable_horizons": (),
            "transient_scaling_state": {"unused_by_fake_session": True},
            "transient_physical_split": analysis.evaluation.artifact_loader._transient_physical_split(split),
            "artifact_provenance": {
                "dataset": {
                    "name": "transient_dataset",
                    "source_dataset_ids": ["transient_dataset"],
                    "source_identities": [deepcopy(split["dataset_identity"]["train"])],
                    "case_ids_by_source": [split["roles"]["id_test"]["case_ids"]],
                    "membership_digests": [split["roles"]["id_test"]["membership_digest"]],
                },
                "lineage": {
                    "stage_identity": {"stage": "b" if arm == "b" else "a", "comparison_arm": arm},
                    "parent_checkpoint": {"source_run_name": "run_a0", "source_checkpoint_sha256": parent},
                    "training_strategy": {
                        "a0": "stage_a_teacher_forced",
                        "a1": "teacher_forced_matched_train_seconds",
                        "b": "rollout",
                    }[arm],
                    "matched_compute_manifest": {"actual": controller},
                },
            },
        }
    )
    if matched:
        receipt = _matched_receipt()
        provenance = frame.attrs["artifact_provenance"]
        provenance["run"] = {"parent_experiment": {"kind": "grouped", "parent_identity_sha256": receipt["parent_identity_sha256"]}}
        matched = provenance["lineage"]["matched_compute_manifest"]
        matched["parent_experiment_sha256"] = receipt["parent_identity_sha256"]
        matched["b_train_budget_receipt"] = receipt
        if arm == "a1":
            controller["matched_receipt_sha256"] = receipt["receipt_sha256"]
    return frame


def _prepared(run_dir: Path, arm: str) -> Any:
    frame = _frame(arm)
    loaded_artifact = analysis.evaluation.artifact_loader.LoadedEvaluationArtifact(
        split_role="eval",
        dataset_name="transient_dataset",
        root=run_dir / "analysis" / "id",
        frame=frame,
        identity_sha256=arm.ljust(64, "a"),
    )
    loaded = analysis.evaluation.artifact_loader.LoadedRunArtifacts(
        task="transient_drying",
        run_name=f"run_{arm}",
        run_dir=run_dir,
        id_artifact=loaded_artifact,
        selected_checkpoint_epoch=3,
        selected_checkpoint_sha256="b" * 64 if arm == "b" else common.serialization.canonical_json_sha256({"arm": arm}),
    )
    return analysis.artifacts.service.PreparedRunArtifacts(
        loaded_run=loaded,
        role_actions={"id": "reused"},
        artifact_device=None,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("completed_stage_epochs", 2, "epoch budget"),
        ("best_within_budget_epoch", 1, "selected checkpoint"),
    ],
)
def test_a0_evidence_rejects_projected_zero_without_terminal_proof(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    """Do not project zero A0 post-handoff compute from incomplete lineage."""
    prepared = _prepared(tmp_path / "a0", "a0")
    frame = prepared.loaded_run.id_artifact.frame
    controller = frame.attrs["artifact_provenance"]["lineage"]["matched_compute_manifest"]["actual"]
    controller[field] = value
    with pytest.raises(ValueError, match=message):
        analysis.evaluation.workflow._transient_arm_evidence(prepared.loaded_run)


def test_workflow_filters_absent_additional_evaluation_and_validates_exact_three_arm_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dispatch transient frames without invented additional data and admit exact A0/A1/B evidence."""
    prepared = {
        (tmp_path / "a0").resolve(): _prepared((tmp_path / "a0").resolve(), "a0"),
        (tmp_path / "a1").resolve(): _prepared((tmp_path / "a1").resolve(), "a1"),
        (tmp_path / "b").resolve(): _prepared((tmp_path / "b").resolve(), "b"),
    }
    observed_frames: dict[str, pd.DataFrame] = {}
    observed_sessions: list[Any] = []
    dataset_calls: list[tuple[str, ...]] = []

    class FakeSession:
        def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
            observed_frames.update(frames)
            observed_sessions.append(self)
            self.frames = tuple(frames)
            self.closed = False

        def dataset_dataframe(self) -> pd.DataFrame:
            dataset_calls.append(self.frames)
            return pd.DataFrame.from_records(
                [
                    {
                        "frame": frame_name,
                        "material_family": "lentil",
                        "mode": "autonomous_full",
                        "requested_horizon": "full",
                        "scope": "cumulative",
                        "normalized_drying_group_macro_rmse": 0.25,
                    }
                    for frame_name in self.frames
                ]
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_or_build_run_artifacts",
        lambda path, **_kwargs: prepared[path],
    )
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.tracking,
        "initialize_wandb",
        lambda *_args, **_kwargs: pytest.fail("notebook Evaluation preparation contacted W&B"),
    )
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(
        analysis.evaluation.panel,
        "build_transient_panel",
        lambda **_kwargs: widgets.Label("transient"),
    )

    workflow = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(
            analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, ("A0", "A1", "B"), strict=True)
        ),
        (
            analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),
            analysis.evaluation.workflow.EvaluationContextSpec("additional_evaluation", "Additional evaluation", "additional_evaluation"),
        ),
    )
    assert workflow.task == "transient_drying"
    assert dataset_calls == []
    assert tuple(context.key for context in workflow.contexts) == ("id",)
    assert all(row["artifact_roles"] == ["id"] for row in workflow.report)
    assert set(observed_frames) == {"A0 ID", "A1 ID", "B ID"}
    assert workflow.transient_lineage is not None
    assert workflow.transient_lineage.primary_comparison == "B_vs_A1"
    assert workflow.transient_lineage.separate_comparison == "B_vs_A0"
    performance = workflow.transient_performance
    assert performance is not None
    assert len(dataset_calls) == 1
    assert set(performance["comparison_arm"]) == {"A0", "A1", "B"}
    assert performance.attrs["primary_comparison"] == "B_vs_A1"
    assert workflow.transient_performance is performance
    assert len(dataset_calls) == 1
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance)
    assert len(figure.axes[0].lines) == 1
    np.testing.assert_allclose(
        np.asarray(figure.axes[0].lines[0].get_xdata(), dtype=np.float64),
        np.asarray([0.0, 10.0, 10.0], dtype=np.float64),
    )
    plt.close(figure)
    workflow.close()
    assert len(observed_sessions) == 1
    assert observed_sessions[0].closed is True


def test_matched_a1_b_lineage_and_plot_use_immutable_b_train_seconds(tmp_path: Path) -> None:
    """Admit receipt-backed A1/B lineage and plot committed training time."""
    parent_identity = "a" * 64
    receipt = matched_budget.build_b_receipt(
        parent_identity_sha256=parent_identity,
        a0_handoff_sha256="c" * 64,
        b_config={
            "task": "transient_drying",
            "run": {"name": "run_b"},
            "training": {
                "matched_a1_b": {"role": "b", "initialization": "a0_best_full_common_fresh_controller"},
                "teacher_handoff": {"source_run_name": "run_a0"},
            },
            "data": {
                "train_dataset": "dataset",
                "additional_evaluation_datasets": [],
                "evaluation_assignments": [],
                "case_partition": {"kind": "exact", "train": ["case-a"], "validation": ["case-b"], "id_test": ["case-c"]},
                "spatial_stride": 1,
            },
            "input_profile": "profile",
            "temporal": {"temporal_conditioning": "time"},
            "model": {"kind": "fno"},
            "evaluation": {"objective": {"id": "objective"}},
        },
        b_summary={
            "status": "completed",
            "terminal_controller": {
                "budget_complete": True,
                "clock_kind": "train_seconds",
                "train_seconds": 12.5,
                "successful_optimizer_steps": 4,
                "post_handoff_optimizer_steps": 4,
            },
            "terminal_epoch": 3,
            "best_epoch": 3,
            "best_checkpoint_sha256": "b" * 64,
            "last_checkpoint_sha256": "d" * 64,
        },
    )
    prepared = tuple(_prepared(tmp_path / arm, arm) for arm in ("a0", "a1", "b"))
    for arm, current in zip(("a0", "a1", "b"), prepared, strict=True):
        provenance = current.loaded_run.id_artifact.frame.attrs["artifact_provenance"]
        provenance["run"] = {
            "parent_experiment": {"kind": "independent"} if arm == "a0" else {"kind": "grouped", "parent_identity_sha256": parent_identity}
        }
        manifest = provenance["lineage"]["matched_compute_manifest"]
        if arm != "a0":
            manifest["parent_experiment_sha256"] = parent_identity
        if arm != "a0":
            controller = manifest["actual"]
            controller.update(
                {
                    "clock_kind": "train_seconds",
                    "budget_control": "train_seconds",
                    "planned_train_seconds": 10.0 if arm == "b" else 12.5,
                    "train_seconds": 12.5 if arm == "b" else 13.0,
                    "matched_receipt_sha256": receipt["receipt_sha256"] if arm == "a1" else None,
                }
            )
            manifest["b_train_budget_receipt"] = receipt
    loaded = tuple(current.loaded_run for current in prepared)
    lineage = analysis.evaluation.workflow._transient_lineage(loaded)
    assert lineage is not None
    assert lineage.a1 is not None
    assert lineage.a1.arm == "A1"
    assert (lineage.primary_comparison, lineage.separate_comparison) == ("B_vs_A1", "B_vs_A0")

    rows = pd.DataFrame.from_records(
        {
            "frame": f"{label} ID",
            "material_family": "lentil",
            "mode": "autonomous_full",
            "requested_horizon": "full",
            "scope": "cumulative",
            "normalized_drying_group_macro_rmse": 0.25,
        }
        for label in ("A0", "A1", "B")
    )
    session = SimpleNamespace(dataset_dataframe=lambda: rows)
    performance = analysis.evaluation.workflow._transient_performance_frame(
        cast("TransientEvaluationSession", session),
        loaded_runs=loaded,
        display_labels=("A0", "A1", "B"),
        lineage=lineage,
    )
    assert performance is not None
    assert set(performance["comparison_arm"]) == {"A0", "A1", "B"}
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance)
    elapsed_seconds = np.asarray(figure.axes[0].lines[0].get_xdata())
    np.testing.assert_allclose(elapsed_seconds, [0.0, 13.0, 12.5])
    assert "[s]" in figure.axes[0].get_xlabel()
    plt.close(figure)

    a1_controller = loaded[1].id_artifact.frame.attrs["artifact_provenance"]["lineage"]["matched_compute_manifest"]["actual"]
    a1_controller["matched_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable B receipt"):
        analysis.evaluation.workflow._transient_lineage(loaded)


def test_single_model_uses_partitioned_case_union_while_comparison_requires_pairs() -> None:
    """Admit disjoint material artifacts for one run without weakening paired models."""
    single_cases = (
        "transient_drying__chickpea__natural__case_0051",
        "transient_drying__lentil__natural__case_0010",
        "transient_drying__kidney_bean__natural__case_0004",
    )

    class FakeSession:
        def __init__(self) -> None:
            self.by_frame = {
                "Single ID": single_cases[:2],
                "Single Additional evaluation": single_cases[2:],
                "UNO ID": single_cases[:2],
                "FNO ID": single_cases[2:],
            }

        def partitioned_case_inventory(self) -> tuple[Any, ...]:
            return tuple(SimpleNamespace(case_id=case_id) for case_id in single_cases)

        def case_ids(self, frame_name: str) -> tuple[str, ...]:
            return self.by_frame[frame_name]

    current = cast("Any", FakeSession())
    union = analysis.evaluation.workflow._transient_selection_case_ids(
        current,
        display_labels=("Single",),
        roles=("id", "additional_evaluation"),
        single_model=True,
    )
    assert union == single_cases
    assert set(single_cases[:2]).isdisjoint(single_cases[2:])

    with pytest.raises(ValueError, match="share no exact case identities"):
        analysis.evaluation.workflow._transient_selection_case_ids(
            current,
            display_labels=("UNO", "FNO"),
            roles=("id",),
            single_model=False,
        )


def test_workspace_wrappers_convert_ordered_paths_and_close_task_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep notebook wrappers thin while preserving order and closing mismatches."""
    calls: list[tuple[tuple[Any, ...], tuple[Any, ...], dict[str, Any]]] = []

    class FakeSession:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = FakeSession()
    workflow = analysis.evaluation.workflow.PreparedEvaluationWorkflow(
        prepared_runs=(cast("PreparedRunArtifacts", SimpleNamespace()), cast("PreparedRunArtifacts", SimpleNamespace())),
        task="transient_drying",
        session=cast("EvaluationSession", session),
        contexts=(),
        panel=widgets.Label("panel"),
        report=(
            {
                "display_label": "first",
                "scientific_run_name": "scientific_first",
                "selected_checkpoint_role": "best",
                "artifact_roles": ["id"],
                "artifact_actions": {"id": "reused"},
            },
            {
                "display_label": "second",
                "scientific_run_name": "scientific_second",
                "selected_checkpoint_role": "best",
                "artifact_roles": ["id"],
                "artifact_actions": {"id": "built"},
            },
        ),
        summary_text="",
    )

    def prepare(selections: tuple[Any, ...], contexts: tuple[Any, ...], **kwargs: Any) -> Any:
        calls.append((selections, contexts, dict(kwargs)))
        return workflow

    monkeypatch.setattr(analysis.evaluation.workflow, "prepare_evaluation_workflow", prepare)
    result = analysis.evaluation.workflow.prepare_model_comparison_evaluation_workspace(
        (tmp_path / "first", tmp_path / "second"),
        labels=("First", "Second"),
        auto_build_missing=False,
        rebuild_incompatible=True,
        device_policy="cpu",
        sections=("summary",),
    )
    assert [selection.label for selection in calls[0][0]] == ["First", "Second"]
    assert [spec.artifact_role for spec in calls[0][1]] == ["id", "additional_evaluation"]
    assert calls[0][2]["auto_build_missing"] is False
    assert calls[0][2]["rebuild_incompatible"] is True
    assert calls[0][2]["device_policy"] == "cpu"
    assert calls[0][2]["sections"] == ("summary",)
    assert "Selected runs: 2" in result.summary_text
    assert "scientific_first" in result.summary_text
    with pytest.raises(ValueError, match="at least two"):
        analysis.evaluation.workflow.prepare_model_comparison_evaluation_workspace((tmp_path / "only",))
    with pytest.raises(TypeError, match="ordered sequence"):
        analysis.evaluation.workflow.prepare_model_comparison_evaluation_workspace(
            (tmp_path / "first", tmp_path / "second"),
            labels="AB",
        )
    with pytest.raises(ValueError, match="unsupported"):
        analysis.evaluation.workflow.prepare_single_model_evaluation_workspace(
            tmp_path / "single",
            expected_task="unknown",
        )

    mismatch = analysis.evaluation.workflow.PreparedEvaluationWorkflow(
        prepared_runs=(cast("PreparedRunArtifacts", SimpleNamespace()),),
        task="steady_flow",
        session=cast("EvaluationSession", session),
        contexts=(),
        panel=widgets.Label("panel"),
        report=workflow.report[:1],
        summary_text="",
    )
    monkeypatch.setattr(analysis.evaluation.workflow, "prepare_evaluation_workflow", lambda *_args, **_kwargs: mismatch)
    with pytest.raises(ValueError, match="expected task"):
        analysis.evaluation.workflow.prepare_single_model_evaluation_workspace(tmp_path / "single")
    assert session.closed is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_name", "different_dataset"),
        ("source_dataset_ids", ("different_source",)),
        ("membership_digests", ("different_membership",)),
        ("case_ids", ("case_0002",)),
    ],
)
def test_shared_role_membership_requires_every_exact_identity_field(field: str, value: Any) -> None:
    """Require dataset, source, digest, and sorted case membership equality."""
    membership = analysis.evaluation.transient_comparison.SharedRoleMembership(
        dataset_name="dataset",
        source_dataset_ids=("source",),
        membership_digests=("digest",),
        case_ids=("case_0001",),
    )
    validator = analysis.evaluation.transient_comparison.validate_shared_role_membership

    assert validator(role="id", memberships=(membership, membership)) == membership
    with pytest.raises(ValueError, match="dataset membership"):
        validator(
            role="id",
            memberships=(membership, replace(membership, **{field: value})),
        )


def test_generic_transient_comparison_rejects_mismatched_role_membership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject generic transient comparisons whose saved case membership differs."""
    prepared = {
        (tmp_path / "first").resolve(): _prepared((tmp_path / "first").resolve(), "a0"),
        (tmp_path / "second").resolve(): _prepared((tmp_path / "second").resolve(), "a1"),
    }
    prepared[(tmp_path / "second").resolve()].loaded_run.id_artifact.frame.attrs["transient_sequence_records"] = (
        SimpleNamespace(case_id="different_case", identity=_frame("a1").attrs["transient_sequence_records"][0].identity),
    )
    closed: list[bool] = []

    class FakeSession:
        def __init__(self, _frames: dict[str, pd.DataFrame]) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    with pytest.raises(ValueError, match="dataset membership"):
        analysis.evaluation.workflow.prepare_evaluation_workflow(
            tuple(analysis.evaluation.workflow.EvaluationRunSelection(path) for path in prepared),
            (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
        )
    assert closed == [True]


def test_physical_projection_preserves_raw_sampling_and_package_associations() -> None:
    """Different expanded item sampling retains the exact physical split without rewriting evidence."""
    split = _split("b")
    split["dataset_identity"]["additional_evaluation"] = [
        {"dataset_id": "additional", "index_digest": "additional_index", "data_contract_digest": "additional_contract"}
    ]
    split["roles"]["additional_evaluation"] = {
        "parts": [{"case_ids": ["additional_case"], "item_ids": ["window"], "membership_digest": "raw_digest"}]
    }
    before = deepcopy(split)
    physical = analysis.evaluation.artifact_loader._transient_physical_split(split)
    assert split == before
    assert physical["additional_evaluation"]["source_dataset_ids"] == ["additional"]
    assert physical["additional_evaluation"]["case_ids_by_source"] == [["additional_case"]]
    assert all("item_ids" not in part for part in physical.values())
    physical["train"]["source_identities"][0]["index_digest"] = "changed_copy"
    assert split == before


def test_strategy_sampling_digests_do_not_change_physical_role_or_lineage() -> None:
    """A0/A1 one-step and B rollout windows compare the same complete physical cases."""
    first = _prepared(Path("a0"), "a0").loaded_run
    second = _prepared(Path("b"), "b").loaded_run
    workflow = analysis.evaluation.workflow
    first_artifact = first.id_artifact
    second_artifact = second.id_artifact
    assert first_artifact is not None
    assert second_artifact is not None
    first_raw = deepcopy(first_artifact.frame.attrs["artifact_provenance"])
    second_raw = deepcopy(second_artifact.frame.attrs["artifact_provenance"])
    assert first_raw["dataset"]["membership_digests"] != second_raw["dataset"]["membership_digests"]
    assert workflow._transient_role_membership(first_artifact) == workflow._transient_role_membership(second_artifact)
    assert workflow._transient_arm_evidence(first).split_id == workflow._transient_arm_evidence(second).split_id
    assert first_artifact.frame.attrs["artifact_provenance"] == first_raw
    assert second_artifact.frame.attrs["artifact_provenance"] == second_raw


@pytest.mark.parametrize("role", ["train", "validation", "id_test", "scaling_train_one_step", "additional_evaluation"])
def test_lineage_rejects_changes_in_every_physical_role(role: str) -> None:
    """Even unchanged evaluated ID cases cannot conceal a different training/validation split."""
    first = _prepared(Path("a0"), "a0").loaded_run
    second = _prepared(Path("b"), "b").loaded_run
    assert second.id_artifact is not None
    physical = second.id_artifact.frame.attrs["transient_physical_split"]
    if role == "additional_evaluation":
        physical[role] = deepcopy(physical["train"])
    physical[role]["case_ids_by_source"] = [["different_case"]]
    workflow = analysis.evaluation.workflow
    assert workflow._transient_arm_evidence(first).split_id != workflow._transient_arm_evidence(second).split_id
    third = _prepared(Path("a1"), "a1").loaded_run
    with pytest.raises(ValueError, match="split_id"):
        workflow._transient_lineage((first, second, third))


@pytest.mark.parametrize("field", ["index_digest", "data_contract_digest"])
def test_shared_role_rejects_different_source_identity(field: str) -> None:
    """The same package name/case cannot conceal a different immutable Source-backed index."""
    first = _prepared(Path("a0"), "a0").loaded_run
    second = _prepared(Path("b"), "b").loaded_run
    assert second.id_artifact is not None
    second.id_artifact.frame.attrs["artifact_provenance"]["dataset"]["source_identities"][0][field] = "different"
    with pytest.raises(ValueError, match="dataset membership"):
        analysis.evaluation.workflow._validate_transient_shared_membership((first, second), roles=("id",))


@pytest.mark.parametrize("field", ["source_identities", "case_ids_by_source"])
def test_missing_physical_role_evidence_fails_closed(field: str) -> None:
    artifact = _prepared(Path("a0"), "a0").loaded_run.id_artifact
    assert artifact is not None
    del artifact.frame.attrs["artifact_provenance"]["dataset"][field]
    with pytest.raises(ValueError, match="physical membership"):
        analysis.evaluation.workflow._transient_role_membership(artifact)


def test_missing_all_role_projection_fails_closed() -> None:
    loaded = _prepared(Path("a0"), "a0").loaded_run
    assert loaded.id_artifact is not None
    del loaded.id_artifact.frame.attrs["transient_physical_split"]
    with pytest.raises(TypeError, match="physical split"):
        analysis.evaluation.workflow._transient_arm_evidence(loaded)


@pytest.mark.parametrize("mutation", ["wrong_dataset", "wrong_length", "duplicate_case", "empty_case"])
def test_physical_projection_rejects_malformed_associations(mutation: str) -> None:
    split = _split("a0")
    if mutation == "wrong_dataset":
        split["dataset_identity"]["train"]["dataset_id"] = None
    elif mutation == "wrong_length":
        split["dataset_identity"]["additional_evaluation"] = [split["dataset_identity"]["train"]]
    elif mutation == "duplicate_case":
        split["roles"]["id_test"]["case_ids"] = ["case_0001", "case_0001"]
    else:
        split["roles"]["id_test"]["case_ids"] = []
    with pytest.raises(ValueError, match="physical membership"):
        analysis.evaluation.artifact_loader._transient_physical_split(split)


@pytest.mark.parametrize("additional", [[], {}, {"parts": [], "unexpected": []}, {"parts": None}])
def test_additional_projection_requires_saved_parts_mapping(additional: Any) -> None:
    """Reject alternate list, missing-key and invalid parts shapes instead of guessing."""
    split = _split("a0")
    split["roles"]["additional_evaluation"] = additional
    with pytest.raises((TypeError, ValueError), match="additional physical split"):
        analysis.evaluation.artifact_loader._transient_physical_split(split)


def test_transient_loader_attaches_existing_admitted_all_role_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real loader boundary carries admitted physical roles without a second run/checkpoint read."""
    split = _split("b")
    admitted = {
        "split_indices": split,
        "normalizer_state": {"scale": "already_admitted"},
        "config": {"task": "transient_drying"},
        "summary": {},
        "selected_checkpoint_role": "best",
        "selected_checkpoint_epoch": 3,
        "selected_checkpoint_sha256": "checkpoint",
    }
    root = tmp_path / "analysis" / "id"
    root.mkdir(parents=True)
    (root / "artifact_provenance.json").write_text("{}")
    transient = analysis.artifacts.transient
    plan = SimpleNamespace(id_role=SimpleNamespace(dataset_name="transient_dataset"), additional_role=None, run_name="run_b")
    monkeypatch.setattr(transient, "transient_artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
    index = SimpleNamespace(frame=pd.DataFrame(), provenance={}, unavailable_horizons=(), identity_sha256="identity")
    monkeypatch.setattr(transient, "validate_transient_role_artifact_index", lambda *_args, **_kwargs: index)
    loaded = analysis.evaluation.artifact_loader._load_admitted_transient_run_artifacts(
        tmp_path,
        admitted,
        artifact_roles=("id",),
        artifact_root_overrides=None,
    )
    assert loaded.id_artifact is not None
    physical = loaded.id_artifact.frame.attrs["transient_physical_split"]
    assert set(physical) == set(split["roles"])
    assert physical["id_test"]["case_ids_by_source"] == [["case_0001"]]
    assert physical["additional_evaluation"]["case_ids_by_source"] == []
    physical["validation"]["case_ids_by_source"][0][0] = "copy_changed"
    assert split["roles"]["validation"]["case_ids"] == ["validation_case"]


def test_general_three_model_comparison_discloses_failed_controlled_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different valid ancestry suppresses controlled claims without hiding ordinary predictions."""
    prepared = {(tmp_path / arm).resolve(): _prepared((tmp_path / arm).resolve(), arm) for arm in ("a0", "a1", "b")}
    frames = prepared[(tmp_path / "a1").resolve()].loaded_run.id_artifact.frame
    frames.attrs["artifact_provenance"]["lineage"]["parent_checkpoint"]["source_checkpoint_sha256"] = "different"

    class FakeSession:
        def __init__(self, _frames: dict[str, pd.DataFrame]) -> None:
            pass

        def close(self) -> None:
            pass

    captured: dict[str, Any] = {}

    def panel(**kwargs: Any) -> widgets.Label:
        captured.update(kwargs)
        return widgets.Label("general comparison")

    monkeypatch.setattr(analysis.artifacts.service, "load_or_build_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", panel)
    result = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path) for path in prepared),
        (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
    )
    assert result.transient_lineage is None
    assert result.transient_performance is None
    assert captured["training_performance"] is None
    assert captured["matched_stage_unavailable_reason"]
    result.close()
