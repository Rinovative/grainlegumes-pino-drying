# ruff: noqa: S101, SLF001
"""Protect task-aware transient workflow dispatch and A0/A+/B lineage admission."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src import analysis

if TYPE_CHECKING:
    from pathlib import Path

    from src.analysis.artifacts.analysis_artifact_service import PreparedRunArtifacts
    from src.analysis.evaluation.evaluation_session import EvaluationSession


def _frame(arm: str, *, parent: str = "parent") -> pd.DataFrame:
    controller: dict[str, Any] = {
        "arm": {"a0": "A0", "a_plus": "A+", "b": "B"}[arm],
        "stage": "stage_b_self_fed" if arm == "b" else "stage_a_teacher_forcing",
        "budget_control": "stage_epochs" if arm == "a0" else "matched_compute",
        "clock_kind": "optimizer_steps",
        "planned_stage_epochs": 3 if arm == "a0" else None,
        "completed_stage_epochs": 3 if arm == "a0" else 0,
        "planned_teacher_forcing_budget_steps": (None if arm == "a0" else 10),
        "successful_optimizer_steps": (3 if arm == "a0" else 10),
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
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "transient_sequence_records": (SimpleNamespace(case_id="case_0001", identity=identity),),
            "transient_unavailable_horizons": (),
            "transient_scaling_state": {"unused_by_fake_session": True},
            "artifact_provenance": {
                "dataset": {
                    "name": "transient_dataset",
                    "source_dataset_ids": ["transient_dataset"],
                    "source_identities": [{"dataset_id": "transient_dataset", "index_digest": "split"}],
                    "membership_digests": ["membership"],
                },
                "lineage": {
                    "stage_identity": {"comparison_arm": arm},
                    "parent_checkpoint": {"source_checkpoint_sha256": parent},
                    "training_strategy": {
                        "a0": "stage_a_teacher_forced",
                        "a_plus": "teacher_forced_continuation",
                        "b": "rollout",
                    }[arm],
                    "matched_compute_manifest": {"actual": controller},
                },
            },
        }
    )
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
        ood_artifact=None,
        selected_checkpoint_epoch=3,
        selected_checkpoint_sha256=arm.ljust(64, "b"),
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


def test_workflow_filters_absent_ood_and_validates_exact_three_arm_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Dispatch transient frames without invented OOD and admit exact A0/A+/B evidence."""
    prepared = {
        (tmp_path / "a0").resolve(): _prepared((tmp_path / "a0").resolve(), "a0"),
        (tmp_path / "a_plus").resolve(): _prepared((tmp_path / "a_plus").resolve(), "a_plus"),
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
            analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, ("A0", "A+", "B"), strict=True)
        ),
        (
            analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),
            analysis.evaluation.workflow.EvaluationContextSpec("ood", "OOD", "ood"),
        ),
    )
    assert workflow.task == "transient_drying"
    assert dataset_calls == []
    assert tuple(context.key for context in workflow.contexts) == ("id",)
    assert all(row["artifact_roles"] == ["id"] for row in workflow.report)
    assert set(observed_frames) == {"A0 ID", "A+ ID", "B ID"}
    assert workflow.transient_lineage is not None
    assert workflow.transient_lineage.primary_comparison == "B_vs_A_plus"
    assert workflow.transient_lineage.separate_comparison == "B_vs_A0"
    performance = workflow.transient_performance
    assert performance is not None
    assert len(dataset_calls) == 1
    assert set(performance["comparison_arm"]) == {"A0", "A+", "B"}
    assert performance.attrs["primary_comparison"] == "B_vs_A_plus"
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
                "Single OOD": single_cases[2:],
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
        roles=("id", "ood"),
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
                "lifecycle_status": "complete",
                "selected_checkpoint_role": "best",
                "artifact_roles": ["id"],
                "artifact_actions": {"id": "reused"},
            },
            {
                "display_label": "second",
                "scientific_run_name": "scientific_second",
                "lifecycle_status": "complete",
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
    assert [spec.artifact_role for spec in calls[0][1]] == ["id", "ood"]
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
        (tmp_path / "second").resolve(): _prepared((tmp_path / "second").resolve(), "a_plus"),
    }
    prepared[(tmp_path / "second").resolve()].loaded_run.id_artifact.frame.attrs["transient_sequence_records"] = (
        SimpleNamespace(case_id="different_case", identity=_frame("a_plus").attrs["transient_sequence_records"][0].identity),
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
