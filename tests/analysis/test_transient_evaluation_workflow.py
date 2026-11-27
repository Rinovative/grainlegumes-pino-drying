"""Protect task-aware transient workflow dispatch and A0/A1/B lineage admission.

Responsibilities:
  - Check transient dispatch, exact role membership, wrapper closure, and A0/A1/B lineage
  - Check that a case title of the built panel ends with the label its run was selected with
  - Check the summary table's dataset-first order, the outlier ranking and table of finite errors, and the compared runs' labels

Design principles:
  - Comparisons require paired scientific identities and immutable B budget evidence

This module does NOT:
  - Train comparison arms or invent missing evaluation roles
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
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
from matplotlib.text import Text
from support.synthetic_task import synthetic_split_payload
from support.transient_evaluation import production_frame, sequence_frame, sequence_record

from src import analysis, common, datasets
from src.experiments import experiments_matched_budget as matched_budget
from src.experiments import experiments_run as run_service
from src.learning.training import learning_training_budget as stage_budget
from src.learning.training import learning_training_history as training_history
from src.learning.transient.learning_transient_handoff import HANDOFF_DIRECTORY

if TYPE_CHECKING:
    from src.analysis.artifacts.analysis_artifact_service import PreparedRunArtifacts
    from src.analysis.evaluation.evaluation_session import EvaluationSession
    from src.analysis.evaluation.evaluation_transient_session import TransientEvaluationSession


def _split(arm: str, *, material_ood: tuple[tuple[str, tuple[str, ...]], ...] = ()) -> dict[str, Any]:
    """Return one admitted transient split; B differs from A0/A1 only in its strategy-owned Train sampling."""
    split = synthetic_split_payload("transient_drying", train_dataset="transient_dataset", material_ood=material_ood)
    if arm == "b":
        rollout = {"mode": "trajectory", "rollout_length": 2}
        split["extension"]["sampling"] = datasets.contracts.transient.TransientSamplingSpec.from_mapping(rollout).as_dict()
    return split


def _arm_transitions(figure: Any) -> list[float]:
    """Return the committed transitions of every arm's point in the first panel of a matched-budget figure, in drawing order."""
    return [float(offset[0]) for collection in figure.axes[0].collections for offset in np.asarray(collection.get_offsets())]


def _budget(unit: str, *, total: int, spent: int) -> dict[str, Any]:
    """Return one completed terminal stage budget in units (A0) or supervised transitions (B, A1)."""
    units, transitions = (spent, 3 * spent) if unit == "units" else (4, spent)
    return {
        "unit": unit,
        "total": total,
        "units": units,
        "supervised_transitions": transitions,
        "train_seconds": 4.0,
        "validated_point": 20,
        "skipped_points": [],
    }


def _finished(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """
    Return B as validate_finished_run admits it: its run directory, summary, admitted terminal budget and last checkpoint.

    B's history in ``run_dir`` holds what its receipt reads: one validation at B's terminal spend after its one update,
    none without one.
    """
    budget = stage_budget.StageBudget(**summary["terminal_budget"])
    run_dir.mkdir(parents=True, exist_ok=True)
    point, spent = float(budget.validated_point), float(budget.spent)
    records = [{"validation/point": point, "budget/supervised_transitions": spent, "global_step": 1.0}] if budget.spent else []
    training_history.history_path(run_dir).write_text(json.dumps({"validations": records}), encoding="utf-8")
    last = {"update_work": [[budget.units, 1, budget.spent]]}
    return {"run_dir": run_dir, "summary": summary, "terminal_budget": budget, "last_checkpoint": last}


# The selected A0's handoff manifest: every B/A1 receipt of its lineage names these bytes' digest.
_A0_HANDOFF = b'{"handoff": "run_a0"}\n'
_A0_HANDOFF_SHA256 = hashlib.sha256(_A0_HANDOFF).hexdigest()
_B_CONFIG: dict[str, Any] = {
    "task": "transient_drying",
    "run": {"name": "run_b"},
    "training": {
        "teacher_handoff": {"source_run_name": "run_a0"},
        "matched_a1_b": {
            "role": "b",
            "receipt_source_run_name": None,
            "initialization": "a0_best_full_common_fresh_controller",
            "lr_scale": 1.0,
        },
    },
    "data": {
        "train_dataset": "transient_dataset",
        "material_ood_datasets": [],
        "evaluation_assignments": [],
        "case_partition": {"kind": "exact", "train": ["train"], "validation": ["val"], "id_test": ["test"]},
        "spatial_stride": 1,
    },
    "input_profile": "canonical_physics_complete",
    "temporal": {"temporal_conditioning": {"kind": "normalized_current_time"}},
    "model": {"kind": "fno", "params": {"hidden_channels": 8}},
    "evaluation": {"objective": {"id": "objective"}},
}
# The one GPU model a B/A1 pair trains on.
_PAIR_SESSIONS = [{"device_type": "cuda", "cuda_device_name": "NVIDIA RTX 6000 Ada Generation"}]
_COMPLETED_B = {
    "status": "completed",
    "runtime_sessions": _PAIR_SESSIONS,
    "terminal_controller": {"post_handoff_optimizer_steps": 10},
    "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
    "terminal_budget": _budget("supervised_transitions", total=10, spent=10),
    "terminal_epoch": 3,
    "best_epoch": 3,
    "best_checkpoint_sha256": "b" * 64,
    "last_checkpoint_sha256": "d" * 64,
}
# B diverged in its first group: no committed transition, no checkpoint; its receipt holds 0 transitions for its A1.
_FIRST_GROUP_DIVERGED_B = {
    "status": "unsuccessful",
    "runtime_sessions": _PAIR_SESSIONS,
    "terminal_controller": {"post_handoff_optimizer_steps": 0},
    "training_outcome": {"status": "diverged", "budget_progress": 0.0, "eligible_points": 0, "divergence": "non-finite loss"},
    "terminal_budget": {**_budget("supervised_transitions", total=10, spent=0), "units": 0, "train_seconds": 0.0, "validated_point": 0},
    "terminal_epoch": 0,
    "best_epoch": None,
    "last_checkpoint_sha256": None,
}


def _matched_receipt(b_summary: dict[str, Any] = _COMPLETED_B, *, handoff: str = _A0_HANDOFF_SHA256) -> dict[str, Any]:
    """Build the supervised-transition receipt of one finished B under the selected A0 handoff."""
    with tempfile.TemporaryDirectory() as run_dir:
        finished_b = _finished(deepcopy(b_summary), Path(run_dir))
        return matched_budget.build_b_receipt(parent_identity_sha256="a" * 64, a0_handoff_sha256=handoff, b_config=_B_CONFIG, finished_b=finished_b)


def _frame(arm: str, *, parent: str = "parent") -> pd.DataFrame:
    matched = arm != "a0"
    controller: dict[str, Any] = {
        "arm": {"a0": "A0", "a1": "A1", "b": "B"}[arm],
        "stage": "stage_b_self_fed" if arm == "b" else "stage_a_teacher_forcing",
        "post_handoff_optimizer_steps": 10 if matched else 3,
        "best_within_budget_epoch": 2,
    }
    budget = _budget("supervised_transitions", total=10, spent=10) if matched else _budget("units", total=12, spent=12)
    identity = {
        "model_kind": "fno",
        "model_parameters": {"hidden_channels": 8},
        "input_profile": "canonical_physics_complete",
        "boundary_representation": "both_interval_endpoints_with_startup_support",
        "scaling_identity": {"semantic_digest": "scale"},
    }
    split = _split(arm)
    id_test = split["roles"]["id_test"]
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "transient_sequence_records": (SimpleNamespace(case_id=id_test["case_ids"][0], identity=identity),),
            "transient_unavailable_horizons": (),
            "transient_scaling_state": {"unused_by_fake_session": True},
            "transient_physical_split": analysis.evaluation.artifact_loader._physical_split(split),
            "artifact_provenance": {
                "dataset": {
                    "name": "transient_dataset",
                    "role": "id",
                    "source_dataset_ids": ["transient_dataset"],
                    "source_identities": [deepcopy(split["dataset_identity"]["train"])],
                    "case_ids_by_source": [list(id_test["case_ids"])],
                    "membership_digests": [id_test["membership_digest"]],
                    "evaluation_assignments": [],
                },
                "lineage": {
                    "stage_identity": {"stage": "b" if arm == "b" else "a", "comparison_arm": arm},
                    "parent_checkpoint": {"source_run_name": "run_a0", "source_checkpoint_sha256": parent},
                    "training_strategy": {
                        "a0": "stage_a_teacher_forced",
                        "a1": "teacher_forced_matched_supervised_transitions",
                        "b": "rollout",
                    }[arm],
                    "matched_compute_manifest": {
                        "actual": controller,
                        "budget": budget,
                        "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
                    },
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
    if arm == "a0":
        manifest = run_dir / HANDOFF_DIRECTORY / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_bytes(_A0_HANDOFF)
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
    )


@pytest.mark.parametrize(
    ("part", "changes"),
    [("budget", {"units": 11, "validated_point": 18}), ("actual", {"best_within_budget_epoch": 1})],
    ids=["incomplete-budget", "other-checkpoint"],
)
def test_a0_evidence_rejects_projected_zero_without_terminal_proof(tmp_path: Path, part: str, changes: dict[str, int]) -> None:
    """Do not project zero A0 post-handoff compute from an unspent unit budget (point 18 of 20) or another selected checkpoint."""
    prepared = _prepared(tmp_path / "a0", "a0")
    frame = prepared.loaded_run.id_artifact.frame
    frame.attrs["artifact_provenance"]["lineage"]["matched_compute_manifest"][part].update(changes)
    with pytest.raises(ValueError, match="selected checkpoint"):
        analysis.evaluation.workflow._transient_arm_evidence(prepared.loaded_run)


def test_workflow_filters_absent_ood_roles_and_validates_exact_three_arm_lineage(
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
                        "contributing_record_count": 1,
                    }
                    for frame_name in self.frames
                ]
            )

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        analysis.artifacts.service,
        "load_prepared_run_artifacts",
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
            analysis.evaluation.workflow.EvaluationContextSpec("material_ood", "Material OOD", "material_ood"),
        ),
    )
    assert workflow.task == "transient_drying"
    assert dataset_calls == []
    assert tuple(context.key for context in workflow.contexts) == ("id",)
    assert all(row["artifact_roles"] == ["id"] for row in workflow.report)
    assert set(observed_frames) == {"A0 · ID", "A1 · ID", "B · ID"}
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
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance, lineage=workflow.transient_lineage)
    np.testing.assert_allclose(_arm_transitions(figure), [0.0, 10.0, 10.0])
    # A0 joins each later arm; A1 and B at the same budget stay unjoined
    assert [np.asarray(line.get_xdata(), dtype=np.float64).tolist() for line in figure.axes[0].lines] == [[0.0, 10.0], [0.0, 10.0]]
    plt.close(figure)
    workflow.close()
    assert len(observed_sessions) == 1
    assert observed_sessions[0].closed is True


def test_matched_a1_b_lineage_and_plot_use_immutable_b_supervised_transitions(tmp_path: Path) -> None:
    """Admit receipt-backed A1/B lineage and plot committed post-handoff supervised transitions."""
    parent_identity = "a" * 64
    receipt = matched_budget.build_b_receipt(
        parent_identity_sha256=parent_identity,
        a0_handoff_sha256=_A0_HANDOFF_SHA256,
        b_config={
            "task": "transient_drying",
            "run": {"name": "run_b"},
            "training": {
                "matched_a1_b": {"role": "b", "initialization": "a0_best_full_common_fresh_controller"},
                "teacher_handoff": {"source_run_name": "run_a0"},
            },
            "data": {
                "train_dataset": "dataset",
                "material_ood_datasets": [],
                "evaluation_assignments": [],
                "case_partition": {"kind": "exact", "train": ["case-a"], "validation": ["case-b"], "id_test": ["case-c"]},
                "spatial_stride": 1,
            },
            "input_profile": "profile",
            "temporal": {"temporal_conditioning": {"kind": "normalized_current_time"}},
            "model": {"kind": "fno"},
            "evaluation": {"objective": {"id": "objective"}},
        },
        finished_b=_finished(
            {
                "status": "completed",
                "runtime_sessions": _PAIR_SESSIONS,
                "terminal_controller": {"post_handoff_optimizer_steps": 4},
                "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
                "terminal_budget": _budget("supervised_transitions", total=10, spent=12),
                "terminal_epoch": 3,
                "best_epoch": 3,
                "best_checkpoint_sha256": "b" * 64,
                "last_checkpoint_sha256": "d" * 64,
            },
            tmp_path / "b_history",
        ),
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
            manifest["actual"]["matched_receipt_sha256"] = receipt["receipt_sha256"] if arm == "a1" else None
            b_budget, a1_budget = _budget("supervised_transitions", total=10, spent=12), _budget("supervised_transitions", total=12, spent=13)
            manifest["budget"] = b_budget if arm == "b" else a1_budget
            manifest["b_train_budget_receipt"] = receipt
    loaded = tuple(current.loaded_run for current in prepared)
    lineage, reason = analysis.evaluation.workflow._transient_lineage(loaded)
    assert reason is None
    assert lineage is not None
    assert lineage.a1 is not None
    assert lineage.a1.arm == "A1"
    assert (lineage.primary_comparison, lineage.separate_comparison) == ("B_vs_A1", "B_vs_A0")

    rows = pd.DataFrame.from_records(
        {
            "frame": f"{label} · ID",
            "material_family": "lentil",
            "mode": mode,
            "requested_horizon": horizon,
            "scope": "cumulative",
            "normalized_drying_group_macro_rmse": 0.25 if records else math.nan,
            "contributing_record_count": records,
        }
        for label in ("A0", "A1", "B")
        # A rolling horizon no case of the material reaches has a row without records: it has no error to set against compute.
        for mode, horizon, records in (("autonomous_full", "full", 1), ("rolling_origin", 4, 0))
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
    assert set(performance["mode"]) == {"autonomous_full"}
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance, lineage=lineage)
    np.testing.assert_allclose(_arm_transitions(figure), [0.0, 13.0, 12.0])  # A0, A1, B at their own committed transitions
    assert "supervised transitions" in figure.axes[0].get_xlabel()
    plt.close(figure)

    a1_controller = loaded[1].id_artifact.frame.attrs["artifact_provenance"]["lineage"]["matched_compute_manifest"]["actual"]
    a1_controller["matched_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable B receipt"):
        analysis.evaluation.workflow._transient_lineage(loaded)


_A1_WITHOUT_RESULT = {
    "status": "unsuccessful",
    "training_outcome": {"status": "no_eligible_checkpoint", "budget_progress": 1.0, "eligible_points": 0, "divergence": None},
}


def _unavailable_runs(monkeypatch: pytest.MonkeyPatch, runs: dict[Path, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> tuple[Path, ...]:
    """Write unsuccessful leaves (summary status and receipt) and admit each as validate_finished_run would: (config, summary, receipt)."""
    for run_dir, (_config, _summary, receipt) in runs.items():
        run_dir.mkdir(parents=True)
        common.serialization.atomic_write_json(common.paths.resolve_run_summary_path(run_dir), {"status": "unsuccessful"})
        common.serialization.atomic_write_json(run_dir / matched_budget.RECEIPT_NAME, receipt)

    def finished(path: Path) -> dict[str, Any]:
        config, summary, _receipt = runs[Path(path)]
        budget = summary.get("terminal_budget")
        return {"config": config, "summary": deepcopy(summary), "terminal_budget": None if budget is None else stage_budget.StageBudget(**budget)}

    monkeypatch.setattr(run_service, "validate_finished_run", finished)
    return tuple(runs)


_A1_CONFIG: dict[str, Any] = {"training": {"matched_a1_b": {"role": "a1"}}}


def test_an_a1_without_an_eligible_checkpoint_enters_the_comparison_from_its_finished_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A0 and B have artifacts; A1 enters from its validated terminal summary with its status and an unavailable result."""
    receipt = _matched_receipt()
    unavailable = _unavailable_runs(monkeypatch, {tmp_path / "a1": (_A1_CONFIG, _A1_WITHOUT_RESULT, receipt)})
    loaded = tuple(_prepared(tmp_path / arm, arm).loaded_run for arm in ("a0", "b"))

    lineage, reason = analysis.evaluation.workflow._transient_lineage(loaded, unavailable_runs=unavailable)

    assert reason is None
    assert lineage is not None
    assert isinstance(lineage.b, analysis.evaluation.transient_comparison.TrainingArmEvidence)
    assert lineage.a1 == analysis.evaluation.transient_comparison.UnavailableArm(
        arm="A1",
        training_status="no_eligible_checkpoint",
        budget_progress=1.0,
        eligible_points=0,
        receipt_sha256=receipt["receipt_sha256"],
        a0_handoff_sha256=_A0_HANDOFF_SHA256,
    )


@pytest.mark.parametrize("b_selected", [False, True], ids=("b-from-a1-receipt", "b-from-its-run"))
def test_both_matched_arms_without_an_eligible_checkpoint_enter_the_comparison_and_its_plot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, b_selected: bool
) -> None:
    """B diverged in its first group, so its 0-second A1 trained nothing: only A0 has an artifact, and the plot names both ends."""
    receipt = _matched_receipt(_FIRST_GROUP_DIVERGED_B)
    runs = {tmp_path / "a1": (_A1_CONFIG, _A1_WITHOUT_RESULT, receipt)}
    if b_selected:
        runs[tmp_path / "b"] = (_B_CONFIG, _FIRST_GROUP_DIVERGED_B, receipt)
    loaded = (_prepared(tmp_path / "a0", "a0").loaded_run,)

    lineage, reason = analysis.evaluation.workflow._transient_lineage(loaded, unavailable_runs=_unavailable_runs(monkeypatch, runs))

    assert reason is None
    assert lineage is not None
    assert (lineage.b.training_status, lineage.b.eligible_points, lineage.a1.training_status) == ("diverged", 0, "no_eligible_checkpoint")
    assert not isinstance(lineage.b, analysis.evaluation.transient_comparison.TrainingArmEvidence)
    assert not isinstance(lineage.a1, analysis.evaluation.transient_comparison.TrainingArmEvidence)
    rows = pd.DataFrame.from_records(
        [
            {
                "frame": "A0 · ID",
                "material_family": "lentil",
                "mode": "autonomous_full",
                "requested_horizon": "full",
                "scope": "cumulative",
                "normalized_drying_group_macro_rmse": 0.25,
                "contributing_record_count": 1,
            }
        ]
    )
    performance = analysis.evaluation.workflow._transient_performance_frame(
        cast("TransientEvaluationSession", SimpleNamespace(dataset_dataframe=lambda: rows)),
        loaded_runs=loaded,
        display_labels=("A0",),
        lineage=lineage,
    )
    assert performance is not None
    figure = analysis.evaluation.plots.transient.plot_training_performance_vs_compute(performance, lineage=lineage)
    note = " ".join(" ".join(text.get_text().split()) for text in figure.artists if isinstance(text, Text))  # the figure title and its note
    assert "B: Diverged at" in note
    assert "A1: No eligible checkpoint at 100 % of its budget, 0 eligible points, no result" in note
    plt.close(figure)


def test_an_unavailable_arm_is_bound_to_its_own_run_and_to_the_selected_a0(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A receipt copied from another B, or a lineage whose receipts name another A0's handoff, is refused."""
    other_b = deepcopy(_B_CONFIG)
    other_b["run"]["name"] = "run_b_other"
    copied = matched_budget.build_b_receipt(
        parent_identity_sha256="a" * 64,
        a0_handoff_sha256=_A0_HANDOFF_SHA256,
        b_config=other_b,
        finished_b=_finished(deepcopy(_FIRST_GROUP_DIVERGED_B), tmp_path / "b_history"),
    )
    runs = {tmp_path / "a1": (_A1_CONFIG, _A1_WITHOUT_RESULT, copied), tmp_path / "b": (_B_CONFIG, _FIRST_GROUP_DIVERGED_B, copied)}
    a0 = (_prepared(tmp_path / "a0", "a0").loaded_run,)
    with pytest.raises(ValueError, match="receipt"):
        analysis.evaluation.workflow._transient_lineage(a0, unavailable_runs=_unavailable_runs(monkeypatch, runs))

    unrelated = _matched_receipt(_FIRST_GROUP_DIVERGED_B, handoff="f" * 64)
    other = tmp_path / "other"
    runs = {other / "a1": (_A1_CONFIG, _A1_WITHOUT_RESULT, unrelated), other / "b": (_B_CONFIG, _FIRST_GROUP_DIVERGED_B, unrelated)}
    with pytest.raises(ValueError, match="handoff"):
        analysis.evaluation.workflow._transient_lineage(a0, unavailable_runs=_unavailable_runs(monkeypatch, runs))


def test_the_workflow_admits_an_unsuccessful_selection_only_into_the_comparison(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unsuccessful A1 selection loads no artifact; the comparison reports its status beside A0 and B."""
    (a1,) = _unavailable_runs(monkeypatch, {(tmp_path / "a1").resolve(): (_A1_CONFIG, _A1_WITHOUT_RESULT, _matched_receipt())})
    prepared = {(tmp_path / arm).resolve(): _prepared((tmp_path / arm).resolve(), arm) for arm in ("a0", "b")}
    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(
        analysis.evaluation.transient_session, "TransientEvaluationSession", lambda frames: SimpleNamespace(frames=frames, close=lambda: None)
    )
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", lambda **_kwargs: widgets.Label("transient"))

    paths = (*prepared, a1)
    workflow = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(paths, ("A0", "B", "A1"), strict=True)),
        (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
    )

    assert workflow.transient_lineage is not None
    assert workflow.transient_lineage.a1.training_status == "no_eligible_checkpoint"
    assert {row["display_label"] for row in workflow.report} == {"A0", "B"}


def test_a_workflow_built_case_title_ends_with_the_label_its_run_was_selected_with(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The workspace's short label reaches the case titles of the real panel; a run selected by its path alone gets no suffix."""

    def prepared(path: Path, **_kwargs: Any) -> PreparedRunArtifacts:
        record = sequence_record("autonomous_full", "full")
        record.identity.update(dataset_role="id", checkpoint_identity={"best_checkpoint_sha256": "d" * 64})
        root = path / "analysis" / "id"
        frame = production_frame((record,), artifact_root=str(root))
        artifact = analysis.evaluation.artifact_loader.LoadedEvaluationArtifact(
            split_role="eval", dataset_name="transient_dataset", root=root, frame=frame, identity_sha256="a" * 64
        )
        loaded = analysis.evaluation.artifact_loader.LoadedRunArtifacts(task="transient_drying", run_name="run", run_dir=path, id_artifact=artifact)
        return analysis.artifacts.service.PreparedRunArtifacts(loaded_run=loaded, role_actions={"id": "reused"})

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", prepared)
    plot, titles = analysis.evaluation.plots.transient.plot_state_maps, []

    def rendered(record: Any, **kwargs: Any) -> Any:
        figure = plot(record, **kwargs)
        titles.append(figure.get_suptitle())
        return figure

    monkeypatch.setattr(analysis.evaluation.plots.transient, "plot_state_maps", rendered)
    for label in ("FNO · B · seed 9", None):
        workflow = analysis.evaluation.workflow.prepare_evaluation_workflow(
            (analysis.evaluation.workflow.EvaluationRunSelection(tmp_path / "run", label=label),),
            (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
            sections=("sample_viewer",),
        )
        analysis.evaluation.panel._LAZY_PANELS[workflow.panel].open()
        workflow.close()
    assert len(titles) == 2
    assert titles[0].endswith(" h (FNO · B · seed 9)")  # the label ends the rendered title, after the time
    assert titles[1].endswith(" h")
    plt.close("all")


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
                "Single · ID": single_cases[:2],
                "Single · Material OOD": single_cases[2:],
                "UNO · ID": single_cases[:2],
                "FNO · ID": single_cases[2:],
            }

        def partitioned_case_inventory(self) -> tuple[Any, ...]:
            return tuple(SimpleNamespace(case_id=case_id) for case_id in single_cases)

        def case_ids(self, frame_name: str) -> tuple[str, ...]:
            return self.by_frame[frame_name]

    current = cast("Any", FakeSession())
    union = analysis.evaluation.workflow._transient_selection_case_ids(
        current,
        display_labels=("Single",),
        roles=("id", "material_ood"),
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
        sections=("summary",),
    )
    assert [selection.label for selection in calls[0][0]] == ["First", "Second"]
    assert [spec.artifact_role for spec in calls[0][1]] == ["id", "material_ood", "parameter_ood"]
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
    """Refuse a generic transient comparison whose only requested role has different saved case membership."""
    prepared = {
        (tmp_path / "first").resolve(): _prepared((tmp_path / "first").resolve(), "a0"),
        (tmp_path / "second").resolve(): _prepared((tmp_path / "second").resolve(), "a1"),
    }
    second_dataset = prepared[(tmp_path / "second").resolve()].loaded_run.id_artifact.frame.attrs["artifact_provenance"]["dataset"]
    second_dataset["case_ids_by_source"] = [["different_case"]]
    created: list[bool] = []

    class FakeSession:
        def __init__(self, _frames: dict[str, pd.DataFrame]) -> None:
            created.append(True)

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    with pytest.raises(ValueError, match=r"No requested Evaluation artifact role.*dataset membership"):
        analysis.evaluation.workflow.prepare_evaluation_workflow(
            tuple(analysis.evaluation.workflow.EvaluationRunSelection(path) for path in prepared),
            (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
        )
    assert created == []  # every role was refused before a session existed, so none has to be closed


def test_a_role_without_shared_membership_is_refused_alone_with_its_reason(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ID role loads for both models while their disjoint parameter-OOD roles are refused, in the workflow and in the panel."""
    prepared = {}
    for arm, case in (("a0", "ood_first"), ("a1", "ood_second")):
        run_dir = (tmp_path / arm).resolve()
        run = _prepared(run_dir, arm)
        ood = pd.DataFrame()
        ood.attrs = deepcopy(run.loaded_run.id_artifact.frame.attrs)
        ood.attrs["artifact_provenance"]["dataset"]["case_ids_by_source"] = [[case]]
        artifact = replace(run.loaded_run.id_artifact, frame=ood, split_role="parameter_ood")
        prepared[run_dir] = replace(run, loaded_run=replace(run.loaded_run, parameter_ood_artifact=artifact))
    frames: dict[str, pd.DataFrame] = {}
    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self, session_frames: dict[str, pd.DataFrame]) -> None:
            frames.update(session_frames)

        def close(self) -> None:
            pass

    def panel(**kwargs: Any) -> widgets.Label:
        captured.update(kwargs)
        return widgets.Label("comparison")

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", panel)
    specs = (
        analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),
        analysis.evaluation.workflow.EvaluationContextSpec("parameter_ood", "Parameter OOD", "parameter_ood"),
    )

    result = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, ("A0", "A1"), strict=True)),
        specs,
        refused_roles={"material_ood": "not ready in A1"},
    )

    assert set(frames) == {"A0 · ID", "A1 · ID"}
    assert tuple(context.key for context in result.contexts) == ("id",)
    assert all(row["artifact_roles"] == ["id"] for row in result.report)
    assert set(result.refused_roles) == {"parameter_ood", "material_ood"}
    assert "dataset membership" in result.refused_roles["parameter_ood"]
    assert result.refused_roles["material_ood"] == "not ready in A1"
    assert dict(captured["refused_roles"]) == dict(result.refused_roles)
    result.close()


def test_a_comparison_without_the_id_role_loads_its_other_role_and_has_no_controlled_stage_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runs that offer only a material-OOD artifact (ID refused before loading) still compare; the lineage, read from ID, says so."""
    prepared = {}
    for arm in ("a0", "a1"):
        run_dir = (tmp_path / arm).resolve()
        run = _prepared(run_dir, arm)
        ood = replace(run.loaded_run.id_artifact, split_role="material_ood")
        prepared[run_dir] = replace(run, loaded_run=replace(run.loaded_run, id_artifact=None, material_ood_artifact=ood))
    frames: dict[str, pd.DataFrame] = {}
    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self, session_frames: dict[str, pd.DataFrame]) -> None:
            frames.update(session_frames)

        def close(self) -> None:
            pass

    def panel(**kwargs: Any) -> widgets.Label:
        captured.update(kwargs)
        return widgets.Label("comparison")

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", panel)

    result = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, ("A0", "A1"), strict=True)),
        (analysis.evaluation.workflow.EvaluationContextSpec("material_ood", "Material OOD", "material_ood"),),
    )

    assert set(frames) == {"A0 · Material OOD", "A1 · Material OOD"}  # the run label and the role's display name, never a raw token
    assert all(row["artifact_roles"] == ["material_ood"] for row in result.report)
    assert result.transient_lineage is None
    assert result.transient_performance is None
    assert captured["training_performance"] is None
    assert "ID role" in str(result.matched_stage_unavailable_reason)
    result.close()


@pytest.mark.parametrize(
    ("leaves", "labels", "run_name"),
    [((("first", "run"), ("second", "run")), ("FNO", "FNO"), None), ((("first", "run"), ("second", "run")), (None, None), "run")],
    ids=["equal explicit labels", "path-only runs of one directory leaf"],
)
def test_runs_of_equal_labels_build_distinct_frames_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, leaves: tuple[tuple[str, str], ...], labels: tuple[str | None, ...], run_name: str | None
) -> None:
    """Two runs shown alike (one explicit label, or one storage alias and run name) keep two frames per role, each run labelled apart."""
    prepared = {}
    for (parent, leaf), arm in zip(leaves, ("a0", "a1"), strict=True):
        run_dir = (tmp_path / parent / leaf).resolve()
        run = _prepared(run_dir, arm)
        prepared[run_dir] = run if run_name is None else replace(run, loaded_run=replace(run.loaded_run, run_name=run_name))
    frames: dict[str, pd.DataFrame] = {}
    captured: dict[str, Any] = {}

    class FakeSession:
        def __init__(self, session_frames: dict[str, pd.DataFrame]) -> None:
            frames.update(session_frames)

        def close(self) -> None:
            pass

    def panel(**kwargs: Any) -> widgets.Label:
        captured.update(kwargs)
        return widgets.Label("comparison")

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", panel)

    result = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, labels, strict=True)),
        (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
    )

    run_labels = [run.label for run in captured["frame_runs"].values()]
    assert len(frames) == 2
    assert set(frames) == set(captured["frame_runs"])
    assert len(set(run_labels)) == 2
    assert all(label.startswith(labels[0] or "run") for label in run_labels)  # the shared label, with the smallest discriminator
    assert len({row["display_label"] for row in result.report}) == 2
    result.close()


def test_physical_projection_drops_only_the_sampling_extension() -> None:
    """The physical split keeps every Dataset identity and role membership, without the task's item expansion."""
    split = _split("b", material_ood=(("additional", ("chickpea__case_0001__00000000000a",)),))
    before = deepcopy(split)
    physical = analysis.evaluation.artifact_loader._physical_split(split)
    assert split == before
    assert set(physical) == {"dataset_identity", "roles"}
    assert physical["dataset_identity"]["material_ood"][0]["dataset_id"] == "additional"
    assert physical["roles"]["material_ood"]["parts"][0]["case_ids"] == ["chickpea__case_0001__00000000000a"]
    physical["dataset_identity"]["train"]["index_digest"] = "changed_copy"
    assert split == before


def test_strategy_sampling_does_not_change_physical_role_or_lineage(tmp_path: Path) -> None:
    """A0/A1 one-step and B rollout sampling compare the same complete physical cases."""
    assert _split("a0")["extension"] != _split("b")["extension"]
    first = _prepared(tmp_path / "a0", "a0").loaded_run
    second = _prepared(tmp_path / "b", "b").loaded_run
    workflow = analysis.evaluation.workflow
    first_artifact = first.id_artifact
    second_artifact = second.id_artifact
    assert first_artifact is not None
    assert second_artifact is not None
    first_raw = deepcopy(first_artifact.frame.attrs["artifact_provenance"])
    assert workflow._role_membership(first_artifact) == workflow._role_membership(second_artifact)
    assert workflow._transient_arm_evidence(first).split_id == workflow._transient_arm_evidence(second).split_id
    assert first_artifact.frame.attrs["artifact_provenance"] == first_raw


@pytest.mark.parametrize("role", ["train", "validation", "id_test", "material_ood"])
def test_lineage_rejects_changes_in_every_physical_role(role: str, tmp_path: Path) -> None:
    """Even unchanged evaluated ID cases cannot conceal a different training/validation split."""
    first = _prepared(tmp_path / "a0", "a0").loaded_run
    second = _prepared(tmp_path / "b", "b").loaded_run
    assert second.id_artifact is not None
    roles = second.id_artifact.frame.attrs["transient_physical_split"]["roles"]
    roles[role] = {"parts": [{"case_ids": ["different_case"]}]} if role == "material_ood" else {"case_ids": ["different_case"]}
    workflow = analysis.evaluation.workflow
    assert workflow._transient_arm_evidence(first).split_id != workflow._transient_arm_evidence(second).split_id
    third = _prepared(tmp_path / "a1", "a1").loaded_run
    lineage, reason = workflow._transient_lineage((first, second, third))
    assert lineage is None
    assert reason is not None
    assert "split_id" in reason


def test_missing_all_role_projection_fails_closed(tmp_path: Path) -> None:
    loaded = _prepared(tmp_path / "a0", "a0").loaded_run
    assert loaded.id_artifact is not None
    del loaded.id_artifact.frame.attrs["transient_physical_split"]
    with pytest.raises(TypeError, match="physical split"):
        analysis.evaluation.workflow._transient_arm_evidence(loaded)


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
    id_role = SimpleNamespace(dataset_name="transient_dataset", root=lambda _run_dir: root)
    plan = SimpleNamespace(id_role=id_role, ood_roles={}, run_name="run_b", role=lambda _split: id_role)
    monkeypatch.setattr(analysis.artifacts.plan, "artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
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
    assert physical["roles"] == split["roles"]
    assert physical["dataset_identity"] == split["dataset_identity"]
    physical["roles"]["validation"]["case_ids"][0] = "copy_changed"
    assert physical["roles"] != split["roles"]


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

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
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


def _history_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, refuse_a1: bool = False
) -> tuple[dict[str, Any], list[tuple[str, str]], dict[str, str], Any]:
    """Prepare an A0/A1 workflow with a fake history reader; return panel kwargs, reads (stage, expected digest), current digests, result."""
    prepared: dict[Path, PreparedRunArtifacts] = {}
    current: dict[str, str] = {}
    for arm in ("a0", "a1"):
        run_dir = (tmp_path / arm).resolve()
        run = _prepared(run_dir, arm)
        prepared[run_dir] = replace(run, loaded_run=replace(run.loaded_run, selected_checkpoint_sha256=arm.ljust(64, "c")))
        current[arm] = arm.ljust(64, "c")
    captured: dict[str, Any] = {}
    reads: list[tuple[str, str]] = []

    class FakeSession:
        def __init__(self, _frames: dict[str, pd.DataFrame]) -> None:
            pass

        def close(self) -> None:
            pass

    def panel(**kwargs: Any) -> widgets.Label:
        captured.update(kwargs)
        return widgets.Label("comparison")

    def read(run_dir: Path, *, expected_checkpoint_sha256: str) -> str:
        reads.append((run_dir.name, expected_checkpoint_sha256))
        if refuse_a1 and run_dir.name == "a1":
            message = "history refused"
            raise RuntimeError(message)
        if current[run_dir.name] != expected_checkpoint_sha256:  # what the real reader checks under its lease
            message = "checkpoint digest differs"
            raise RuntimeError(message)
        return f"history of {run_dir.name}"

    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    monkeypatch.setattr(analysis.evaluation.transient_session, "TransientEvaluationSession", FakeSession)
    monkeypatch.setattr(analysis.evaluation.panel, "build_transient_panel", panel)
    monkeypatch.setattr(analysis.evaluation.run_discovery, "stage_history", read)
    result = analysis.evaluation.workflow.prepare_evaluation_workflow(
        tuple(analysis.evaluation.workflow.EvaluationRunSelection(path, label=label) for path, label in zip(prepared, ("A0", "A1"), strict=True)),
        (analysis.evaluation.workflow.EvaluationContextSpec("id", "ID", "id"),),
    )
    return captured, reads, current, result


def test_the_training_histories_are_read_only_when_the_panel_asks_and_a_refused_history_shows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The panel gets a loader of every loaded stage's history by label, each read bound to the loaded checkpoint; a refusal is not caught."""
    captured, reads, current, result = _history_workflow(tmp_path, monkeypatch, refuse_a1=True)

    assert reads == []
    with pytest.raises(RuntimeError, match="history refused"):
        captured["training_histories"]()
    assert reads == [("a0", current["a0"]), ("a1", current["a1"])]
    result.close()


def test_every_opening_reads_the_histories_again_so_a_replaced_run_is_seen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two openings read every stage twice; a stage whose checkpoint digest changed in between fails the second opening."""
    captured, reads, current, result = _history_workflow(tmp_path, monkeypatch)

    assert captured["training_histories"]() == {"A0": "history of a0", "A1": "history of a1"}
    current["a1"] = "e" * 64
    with pytest.raises(RuntimeError, match="digest differs"):
        captured["training_histories"]()
    assert [name for name, _digest in reads] == ["a0", "a1", "a0", "a1"]
    result.close()


def test_the_transient_summary_table_lists_materials_by_display_label_and_horizons_ascending() -> None:
    """Lentil enters first and horizon 2 before 1: the table lists Chickpea first and each material's horizons 1, 2 and the unsupported 4."""
    records = tuple(
        sequence_record("rolling_origin", horizon, case_id=f"{material}-case", material_family=material)
        for material in ("lentil", "chickpea")
        for horizon in (2, 1)
    )
    evaluation = analysis.evaluation.transient_session.TransientEvaluationSession({"id": sequence_frame(records)})
    table = analysis.evaluation.panel._transient_summary_table(evaluation)
    rows = table.loc[table["scope"] == "cumulative", ["material_family", "requested_horizon"]]
    expected = [(material, horizon) for material in ("chickpea", "lentil") for horizon in (1, 2, 4)]
    assert list(rows.itertuples(index=False, name=None)) == expected
    evaluation.close()


def test_the_transient_summary_table_orders_by_dataset_before_the_frame_order(tmp_path: Path) -> None:
    """Runs A and B each hold an ID and a Material OOD frame: both ID frames come first, A before B, then both Material OOD frames."""
    frames = {}
    for run in ("A", "B"):
        for role, material in (("id", "lentil"), ("material_ood", "chickpea")):
            record = sequence_record("autonomous_full", "full", case_id=f"{material}-case", material_family=material)
            record.identity["checkpoint_identity"] = {"best_checkpoint_sha256": run.lower() * 64}
            frames[f"{run} {role}"] = production_frame((record,), artifact_root=str(tmp_path / f"{run}-{role}"), dataset_role=role)
            frames[f"{run} {role}"].attrs["artifact_provenance"]["run"]["best_checkpoint_sha256"] = run.lower() * 64
    evaluation = analysis.evaluation.transient_session.TransientEvaluationSession(frames)
    table = analysis.evaluation.panel._transient_summary_table(evaluation)
    assert list(dict.fromkeys(table["frame"])) == ["A id", "B id", "A material_ood", "B material_ood"]
    evaluation.close()


def test_the_outlier_view_opens_on_rank_one_of_its_ranking_field_whatever_case_another_view_chose(tmp_path: Path) -> None:
    """The sample viewer chose case «a», the smallest error: the outlier focus still opens on «b», the largest T error, and names it so."""
    records = tuple(
        sequence_record("autonomous_full", "full", case_id=case_id, offset=offset) for case_id, offset in (("a", 1.0), ("b", 5.0), ("c", 3.0))
    )
    evaluation = analysis.evaluation.transient_session.TransientEvaluationSession({"id": production_frame(records, artifact_root=str(tmp_path))})
    runs = (SimpleNamespace(run_dir=(tmp_path / "run").resolve()),)
    group = analysis.evaluation.run_discovery.EvaluationRunGroup("a" * 64, "runs", "transient_drying", cast("Any", runs))
    catalog = analysis.evaluation.run_discovery.EvaluationRunCatalog(cast("Any", runs), (group,), (), {}, {}, {}, {})
    state = analysis.evaluation.selection.EvaluationSelectionState(catalog, comparison=False)
    panel = analysis.evaluation.panel
    temperature = SimpleNamespace(key="T", stored=True)
    panel._publish_transient_capabilities(state, (temperature,), ("a", "b", "c"), channels=("T",), case_id="a")
    ranks: dict[tuple[str, str], tuple[str, int]] = {}
    focus = panel._CaseFocus(
        evaluation,
        panel._transient_inventory(evaluation, comparison=False, selection_state=state),
        inventory=panel.dataset_cuts.transient_inventory(evaluation, refused={}),
        multiple=False,
        comparison=False,
        selection_state=state,
        status=analysis.ui.controls.StatusLine(),
        on_change=lambda: None,
        order=panel._transient_outlier_order(evaluation, fields=(temperature,), ranks=ranks),
    )
    focus.pull()
    assert focus.entry.case_id == "b"  # the shared «a» stands third in the ranking
    assert panel._transient_sample_title(focus.entry, outlier=True, rank=ranks[focus.entry.frame_name, "b"]).startswith("Largest $T$ error in")
    evaluation.close()


def test_the_outlier_ranking_ranks_finite_errors_only_and_titles_its_fallback_unranked() -> None:
    """T is unavailable for every case and phi for «c»: phi ranks «b» before «a»; without any finite error the cases stay unranked."""
    panel = analysis.evaluation.panel
    frame = pd.DataFrame({"frame": "F", "case_id": ["a", "b", "c"], "requested_horizon": "full", "scope": "cumulative"})
    frame = frame.assign(normalized_rmse_T=math.nan, normalized_rmse_phi=[0.1, 0.3, math.nan])
    fields = (SimpleNamespace(key="T", stored=True), SimpleNamespace(key="phi", stored=True))
    entries = tuple(SimpleNamespace(frame_name="F", case_id=case_id) for case_id in ("a", "b", "c"))
    ranks: dict[tuple[str, str], tuple[str, int]] = {}
    cases = (
        (frame, ["b", "a"], {("F", "b"): ("phi", 1), ("F", "a"): ("phi", 2)}),
        (frame.assign(normalized_rmse_phi=math.nan), ["a", "b", "c"], {}),  # the inventory order
    )
    for rows, expected, ranked in cases:
        session = cast("Any", SimpleNamespace(case_dataframe=lambda rows=rows, **_kwargs: rows))
        assert [entry.case_id for entry in panel._transient_outlier_order(session, fields=fields, ranks=ranks)(entries)] == expected
        assert ranks == ranked
    entry = SimpleNamespace(material_family="lentil", dataset_role="id", evaluation_assignment=None, case_label="case_a")
    assert panel._transient_sample_title(entry, outlier=True, ranked=False).startswith("Unranked case in")


def test_the_outlier_table_ranks_finite_errors_only_and_stays_explicitly_empty_without_any() -> None:
    """T is NaN for «c» and infinite for «d»: only «b» and «a» rank, fewer than the table holds; without a finite error it is empty."""
    panel = analysis.evaluation.panel
    frame = pd.DataFrame(
        {
            "frame": "F",
            "case_id": ["a", "b", "c", "d"],
            "requested_horizon": "full",
            "scope": "cumulative",
            "material_family": "lentil",
            "declared_evaluation_role": "id",
            "normalized_rmse_T": [0.1, 0.3, math.nan, math.inf],
        }
    )
    fields = (SimpleNamespace(key="T", stored=True, label="T", unit="K"),)
    for rows, expected in ((frame, ["b", "a"]), (frame.assign(normalized_rmse_T=[math.nan, math.inf, math.nan, -math.inf]), [])):
        session = cast("Any", SimpleNamespace(case_dataframe=lambda rows=rows, **_kwargs: rows, case_inventory=tuple))
        table = panel._transient_outlier_table(session, fields=fields)
        assert (table["Exact case"].tolist(), table["Rank"].tolist()) == (expected, list(range(1, len(expected) + 1)))


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        (("b-a1__rollout-32",) * 3, ("FNO · B · seed 9", "RNO · B · seed 9", "U-NO · B · seed 9")),
        (
            ("b-a1__rollout-32", "b-a1__rollout-32", "b-a1__rollout-16"),
            ("FNO · B · seed 9 · rollout-32", "RNO · B · seed 9 · rollout-32", "U-NO · B · seed 9 · rollout-16"),
        ),
    ],
    ids=["equal-training", "differing-training"],
)
def test_a_comparison_names_its_runs_by_labels_unique_over_the_compared_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pairs: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    """
    FNO and RNO collide with smoke runs in the catalog, U-NO does not: the compared runs carry one schema, no facet for some only.

    A training setting the compared runs differ in is shown for each, though the architectures alone separate them, and a load
    failure lists the runs by the same labels.
    """
    workspace = analysis.evaluation.workspace
    runs = tuple(
        SimpleNamespace(
            model_family=family,
            model_label=family,
            stage="B",
            seed=9,
            run_dir=tmp_path / family,
            pair_label=pair,
            config_label=f"{family}/config",
            run_name=f"{family}-b",
            dataset_id="demo",
            task="transient_drying",
        )
        for family, pair in zip(("fno", "rno", "uno"), pairs, strict=True)
    )
    controller = object.__new__(workspace._EvaluationWorkspaceController)
    controller._labels = {run.run_dir: f"{label} · rollout-32" for run, label in zip(runs, ("FNO · B · seed 9", "RNO · B · seed 9"), strict=False)}
    controller._labels[runs[2].run_dir] = "U-NO · B · seed 9"
    placeholder = cast("Any", None)
    controller.state, controller.compute_preference, controller.widths = placeholder, cast("Any", "cpu"), placeholder
    monkeypatch.setattr(controller, "_unshared_roles", lambda *_args: {})
    captured: dict[str, Any] = {}

    def prepare(_run_dirs: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        message = "stop after the labels"
        raise RuntimeError(message)

    monkeypatch.setattr(workspace.workflow, "prepare_model_comparison_evaluation_workspace", prepare)
    outcome = controller._load_comparison(cast("Any", runs), ("id",))
    assert "stop after the labels" in outcome.details
    assert captured["labels"] == expected
    assert all(f"<li>{label}: " in outcome.details for label in expected)
