# ruff: noqa: S101, PLR0915, PLR2004, PERF401, SLF001
"""Protect transient Evaluation sequence, rollout, and persistence contracts."""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src import common, datasets
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.analysis.artifacts import analysis_artifact_performance as artifact_performance
from src.analysis.artifacts import analysis_artifact_service as artifact_service
from src.analysis.artifacts import analysis_artifact_transient as artifact_generator
from src.analysis.evaluation import evaluation_dataframe
from src.analysis.evaluation import evaluation_transient_artifact as artifact
from src.analysis.evaluation import evaluation_transient_rollout as rollout
from src.analysis.evaluation import evaluation_transient_session as transient_session
from src.analysis.presentation import analysis_presentation_curated as curated
from src.learning.inference import learning_inference_transient as inference


def _items() -> list[dict[str, Any]]:
    states: list[np.ndarray] = []
    for index in range(4):
        state = np.zeros((4, 2, 2), dtype=np.float32)
        state[0] = 300.0 + index
        state[1] = 0.5
        state[2] = 12.0 - index
        state[3] = 10.0 - index
        states.append(state)
    static = np.zeros((7, 2, 2), dtype=np.float32)
    static[0], static[1] = np.meshgrid(
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    )
    static[2] = 1.0
    static[4] = 101325.0
    static[5] = 0.4
    static[6] = 100.0
    scalars = np.asarray([1.0, 0.5, 0.25, 1.0, 0.01, 1.0, 0.2, 1000.0], dtype=np.float32)
    result: list[dict[str, Any]] = []
    for index in range(3):
        result.append(
            {
                "state": torch.from_numpy(states[index]),
                "static": torch.from_numpy(static),
                "boundary": torch.from_numpy(np.arange(9, dtype=np.float32) + index),
                "scalars": torch.from_numpy(scalars),
                "time": {
                    "t_n": torch.tensor(float(index)),
                    "t_n_plus_1": torch.tensor(float(index + 1)),
                    "dt": torch.tensor(1.0),
                },
                "target": torch.from_numpy(states[index + 1] - states[index]),
                "metadata": {
                    "dataset_id": "transient_dataset",
                    "dataset_name": "transient_dataset",
                    "sample_id": f"case__step_{index:04d}",
                    "sample_mode": "one_step_transition",
                    "rollout_length": 1,
                    "simulation_case_id": "case",
                    "case_input_id": "case-input",
                    "source_batch_id": "batch",
                    "source_simulation_profile": "transient_drying",
                    "material_family": "lentil",
                    "evaluation_regime": "id",
                    "split": "id_test",
                    "time_index_n": index,
                    "time_index_n_plus_1": index + 1,
                    "sequence_length": 4,
                    "stored_state_count": 4,
                    "has_exact_stop_state": False,
                    "t_stop_exact": 3.0,
                },
            }
        )
    return result


def _identity(*, model_kind: str = "rno") -> dict[str, Any]:
    return {
        "case_id": "case",
        "dataset_identity": {"dataset_id": "transient_dataset", "index_digest": "a" * 64},
        "dataset_role": "id",
        "material_family": "lentil",
        "simulation_identity": {"simulation_case_id": "case", "source_batch_id": "batch"},
        "model_kind": model_kind,
        "model_parameters": {"hidden_channels": 8},
        "checkpoint_identity": {"sha256": "b" * 64},
        "input_profile": "canonical_physics_complete_v1",
        "coordinate_policy": "explicit_x_y",
        "boundary_representation": "both_interval_endpoints_with_startup_support",
        "scaling_identity": {"semantic_digest": "c" * 64},
        "training_airflow_source": "comsol_reference",
        "inference_airflow_source": "comsol_reference",
        "stage_identity": {"stage": "B"},
        "training_strategy": "rollout",
        "curriculum_identity": {"active_stage": 0},
        "parent_checkpoint": {"source_checkpoint_sha256": "e" * 64},
        "stage_a_handoff": {"source_checkpoint_sha256": "e" * 64},
        "matched_compute_manifest": {"planned": {}, "actual": {}},
        "dataset_backend": "canonical_hdf5",
        "pt_payload_identity": None,
        "evaluation_config_identity": "d" * 64,
        "timing_evidence_identity": "f" * 64,
    }


def _context() -> inference.TransientInferenceContext:
    return inference.TransientInferenceContext(
        model=nn.Identity(),
        tensorizer=cast("Any", object()),
        scaling=cast(
            "Any",
            SimpleNamespace(
                spatial_shape=(2, 2),
                horizon=168.0,
                device=torch.device("cpu"),
            ),
        ),
        device=torch.device("cpu"),
        model_kind="rno",
        precision="float32",
    )


def _qualified_identity(case_id: str, material: str) -> dict[str, Any]:
    """Return one minimal case identity for fail-fast membership coverage."""
    return {
        "case_id": case_id,
        "material_family": material,
        "simulation_identity": {"package_case_id": case_id},
    }


def test_fail_fast_membership_admits_colliding_local_case_labels_by_qualified_identity() -> None:
    """Keep equal material-local labels distinct through qualified Dataset identities."""
    planned = (
        ("transient_drying__chickpea__natural__case_0001", "chickpea"),
        ("transient_drying__chickpea__natural__case_0002", "chickpea"),
        ("transient_drying__lentil__natural__case_0001", "lentil"),
        ("transient_drying__lentil__natural__case_0002", "lentil"),
    )
    produced = tuple(_qualified_identity(case_id, material) for case_id, material in planned)

    admitted = artifact_generator._qualified_case_membership(
        planned=planned,
        produced=produced,
    )

    assert admitted == frozenset(planned)
    assert len(admitted) == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "missing="),
        ("extra", "unexpected="),
        ("duplicate", "duplicate qualified"),
        ("wrong_material", "missing="),
        ("unqualified", "lost its qualified"),
    ],
)
def test_fail_fast_membership_rejects_inexact_qualified_cases(
    mutation: str,
    message: str,
) -> None:
    """Reject missing, extra, duplicate, or incorrectly qualified produced cases."""
    planned = (
        ("transient_drying__chickpea__natural__case_0001", "chickpea"),
        ("transient_drying__chickpea__natural__case_0002", "chickpea"),
        ("transient_drying__lentil__natural__case_0001", "lentil"),
        ("transient_drying__lentil__natural__case_0002", "lentil"),
    )
    produced = [_qualified_identity(case_id, material) for case_id, material in planned]
    if mutation == "missing":
        produced.pop()
    elif mutation == "extra":
        produced.append(
            _qualified_identity(
                "transient_drying__kidney_bean__natural__case_0001",
                "kidney_bean",
            )
        )
    elif mutation == "duplicate":
        produced.append(dict(produced[0]))
    elif mutation == "wrong_material":
        produced[0] = {**produced[0], "material_family": "lentil"}
    else:
        produced[0] = {
            **produced[0],
            "simulation_identity": {"package_case_id": "case_0001"},
        }

    with pytest.raises(ValueError, match=message):
        artifact_generator._qualified_case_membership(
            planned=planned,
            produced=produced,
        )


def test_base_identity_distinguishes_qualified_package_and_local_generation_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist qualified Dataset membership without losing local Generation lineage."""
    qualified = "transient_drying__chickpea__natural__case_0001"
    case = SimpleNamespace(
        case_id=qualified,
        metadata={
            "dataset_id": "transient_dataset",
            "material_family": "chickpea",
            "source_batch_id": "batch",
        },
    )
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset", "index_digest": "a" * 64},),
        case_ids_by_source=((qualified,),),
        membership_digests=("b" * 64,),
    )
    profile = SimpleNamespace(coordinate_policy="explicit_x_y")
    task = SimpleNamespace(
        input_profile=lambda _profile_id: profile,
        training_airflow_source="comsol_reference",
    )
    scaling = SimpleNamespace(
        digest="c" * 64,
        scale_mode="state_std",
        spatial_shape=(2, 2),
        horizon=168.0,
        tensorizer=SimpleNamespace(as_dict=lambda: {"input_profile": "profile"}),
    )
    monkeypatch.setattr(
        artifact_generator.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: task,
    )
    monkeypatch.setattr(
        artifact_generator.TransientScalingArtifact,
        "from_state_dict",
        lambda _state: scaling,
    )
    monkeypatch.setattr(
        artifact_generator,
        "_lineage",
        lambda _completed: {
            "stage_identity": {"stage": "A0"},
            "training_strategy": "teacher_forced",
            "curriculum_identity": {"active_stage": 0},
            "parent_checkpoint": {"source_checkpoint_sha256": "d" * 64},
            "stage_a_handoff": {"source_checkpoint_sha256": "d" * 64},
            "matched_compute_manifest": {"planned": {}, "actual": {}},
        },
    )
    monkeypatch.setattr(
        artifact_generator.sequence_artifact,
        "evaluation_protocol_identity",
        lambda _evaluation: "e" * 64,
    )
    completed = {
        "config": {
            "input_profile": "profile",
            "model": {"kind": "uno", "params": {"hidden_channels": 8}},
            "evaluation": {},
        },
        "normalizer_state": {},
        "normalizer_sha256": "f" * 64,
        "checkpoint_identity": {"epoch": 1},
        "selected_checkpoint_sha256": "1" * 64,
        "selected_checkpoint_epoch": 1,
    }
    canonical = {
        "source": {
            "simulation_case_id": "2" * 64,
            "case_input_id": "3" * 64,
            "case_id": "case_0001",
            "simulation_profile": "transient_drying",
            "template_sha256": "4" * 64,
        }
    }

    identity = artifact_generator._base_identity(
        completed=completed,
        role=role,
        case=cast("Any", case),
        canonical=canonical,
        dataset_backend="canonical_hdf5",
        pt_identity=None,
        runtime_identity={"scientific_solver_seconds": 1.0},
    )

    assert identity["case_id"] == qualified
    assert identity["simulation_identity"]["package_case_id"] == qualified
    assert identity["simulation_identity"]["generation_case_id"] == "case_0001"


def test_transient_artifact_reuses_exact_composite_generation_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Interpret a composite-backed Dataset case without requiring a false parent terminal."""
    package_case_id = "transient_drying__lentil__natural__case_0001"
    batch_evidence = object()
    case_evidence = object()
    prepared = SimpleNamespace(
        candidates=[
            {
                "package_case_id": package_case_id,
                "terminal_evidence": batch_evidence,
                "case_evidence": case_evidence,
            }
        ]
    )
    monkeypatch.setattr(
        artifact_generator.package_builder,
        "admit_package_composite_sources",
        lambda dataset_id, **_kwargs: prepared if dataset_id == "transient_dataset" else None,
    )
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({},),
        case_ids_by_source=((package_case_id,),),
        membership_digests=("a" * 64,),
    )
    sources = artifact_generator._generation_case_sources(role, storage_root=tmp_path)
    assert sources == {package_case_id: (batch_evidence, case_evidence)}

    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    reference = np.arange(3 * len(artifact.STATE_ORDER) * 4, dtype=np.float32).reshape(3, len(artifact.STATE_ORDER), 2, 2)
    static = np.arange(len(artifact.STATIC_ORDER) * 4, dtype=np.float32).reshape(len(artifact.STATIC_ORDER), 2, 2)
    boundary = np.arange(2 * len(artifact.BOUNDARY_ORDER), dtype=np.float32).reshape(2, len(artifact.BOUNDARY_ORDER))
    scalars = np.arange(len(artifact.SCALAR_ORDER), dtype=np.float32)
    canonical = {
        "state_trajectories": {name: reference[:, index] for index, name in enumerate(artifact.STATE_ORDER)},
        "static_fields": {name: static[index] for index, name in enumerate(artifact.STATIC_ORDER)},
        "boundary_intervals": {name: boundary[:, index] for index, name in enumerate(artifact.BOUNDARY_ORDER)},
        "scalar_conditioning": {name: float(scalars[index]) for index, name in enumerate(artifact.SCALAR_ORDER)},
        "time": {"regular_state_hours": times},
        "runtime": {"scientific_solver_seconds": 1.0},
    }

    def interpret(batch: Any, case: Any, **_kwargs: Any) -> dict[str, Any]:
        assert batch is batch_evidence
        assert case is case_evidence
        return canonical

    monkeypatch.setattr(artifact_generator.generated_batch, "interpret_generated_transient_case", interpret)
    monkeypatch.setattr(
        artifact_generator,
        "_terminal_case",
        lambda **_kwargs: pytest.fail("composite source fell back to terminal parent admission"),
    )
    evaluation_case = rollout.TransientEvaluationCase(
        case_id="simulation-case",
        dataset_role="id",
        physical_times=times,
        reference_states=reference,
        static_conditioning=static,
        boundary_conditioning=boundary,
        scalar_conditioning=scalars,
        spatial_mask=np.ones((2, 2), dtype=np.float32),
        metadata={},
    )
    admitted, runtime = artifact_generator._canonical_case_evidence(
        evaluation_case,
        package_case_id=package_case_id,
        case_record={},
        storage_root=tmp_path,
        generation_sources=sources,
    )
    assert admitted is canonical
    assert runtime == canonical["runtime"]


def test_transient_artifact_resolves_terminal_storage_name_from_source_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Distinguish persisted batch identity from its purpose-qualified storage name."""
    case = SimpleNamespace(case_id="case_0001", simulation_case_id="simulation-case")
    batch = SimpleNamespace(batch_id="canonical-batch-id", cases=(case,))
    admitted_names: list[str] = []

    def admit(name: str, **_kwargs: Any) -> Any:
        admitted_names.append(name)
        return batch

    monkeypatch.setattr(artifact_generator.generation.runtime, "admit_terminal_batch", admit)
    admitted_batch, admitted_case = artifact_generator._terminal_case(
        package_case_id="package-case",
        case_record={
            "source_batch_id": "canonical-batch-id",
            "source_relative_path": "01_generation/processed/purpose-qualified-storage/case_0001/case.h5",
            "simulation_case_id": "simulation-case",
        },
        storage_root=tmp_path,
    )
    assert admitted_names == ["purpose-qualified-storage"]
    assert admitted_batch is batch
    assert admitted_case is case


def test_transient_artifact_projects_generation_target_completion() -> None:
    """Exclude canonical moisture context from strict Generation target evidence."""
    generation_completion = {
        "physical_duration_hours": 3.5,
        "time_to_target_hours": 3.5,
        "target_reached": True,
        "right_censored": False,
        "final_wet_fraction": 0.25,
        "target_wet_fraction_limit": 0.5,
        "physical_duration_availability": "available",
        "target_wet_fraction_limit_availability": "available",
    }
    canonical_completion = {
        **generation_completion,
        "final_bulk_moisture_wb": 0.11,
        "target_moisture_wb": 0.12,
    }

    with pytest.raises(ValueError, match="Generation timing contract"):
        rollout._admit_canonical_target_completion(  # pyright: ignore[reportPrivateUsage]
            canonical_completion,
            expected_limit=0.5,
        )

    projected = artifact_generator._generation_target_completion(canonical_completion)

    assert projected == generation_completion
    admitted = rollout._admit_canonical_target_completion(  # pyright: ignore[reportPrivateUsage]
        projected,
        expected_limit=0.5,
    )
    assert admitted.physical_duration_hours == pytest.approx(3.5)
    assert canonical_completion["final_bulk_moisture_wb"] == pytest.approx(0.11)
    assert artifact_generator._generation_target_completion(generation_completion) == generation_completion

    roundoff_completion = {
        **generation_completion,
        "time_to_target_hours": None,
        "target_reached": False,
        "right_censored": True,
        "final_wet_fraction": 1.0 + 4.0 * math.ulp(1.0),
    }
    admitted_roundoff = rollout._admit_canonical_target_completion(  # pyright: ignore[reportPrivateUsage]
        roundoff_completion,
        expected_limit=0.5,
    )
    assert admitted_roundoff.final_wet_fraction == 1.0 + 4.0 * math.ulp(1.0)
    for invalid_fraction in (1.0 + 5.0 * math.ulp(1.0), 1.01):
        with pytest.raises(ValueError, match="physical range"):
            rollout._admit_canonical_target_completion(  # pyright: ignore[reportPrivateUsage]
                {**roundoff_completion, "final_wet_fraction": invalid_fraction},
                expected_limit=0.5,
            )


def test_artifact_metric_normalization_uses_scaler_runtime_placement() -> None:
    """Move portable record arrays to the scaler contract and return CPU arrays."""

    class CapturingScaling:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self) -> None:
            self.observed: list[tuple[torch.device, torch.dtype]] = []

        def encode_state(self, value: torch.Tensor) -> torch.Tensor:
            self.observed.append((value.device, value.dtype))
            return value + 1.0

    scaling = CapturingScaling()
    normalized = artifact_generator._normalized_metric_state(
        np.zeros((2, 4, 1, 1), dtype=np.float64),
        scaling=cast("Any", scaling),
    )

    assert scaling.observed == [(scaling.device, scaling.dtype)]
    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == np.float32
    assert np.all(normalized == 1.0)


def test_artifact_metric_statistics_reuse_cumulative_normalization_and_spatial_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive endpoint metrics by slicing one normalized trajectory and one weight grid."""
    normalization_shapes: list[tuple[int, ...]] = []
    weight_calls: list[np.ndarray] = []

    def normalize(value: np.ndarray, *, scaling: object) -> np.ndarray:
        del scaling
        normalization_shapes.append(value.shape)
        return np.asarray(value, dtype=np.float32)

    def weights(mask: np.ndarray) -> np.ndarray:
        weight_calls.append(mask)
        return np.ones(mask.shape, dtype=np.float64)

    monkeypatch.setattr(artifact_generator, "_normalized_metric_state", normalize)
    monkeypatch.setattr(artifact_generator.transient_metrics, "trapezoidal_cell_weights", weights)
    predicted = np.zeros((3, 4, 2, 2), dtype=np.float32)
    predicted[:, 0] = 300.0
    predicted[:, 1] = 0.5
    predicted[:, 2:] = 1.0
    reference = predicted.copy()
    static = np.ones((7, 2, 2), dtype=np.float32)
    record = SimpleNamespace(
        predicted_states=predicted,
        reference_states=reference,
        spatial_mask=np.ones((2, 2), dtype=bool),
        scalar_conditioning=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        static_conditioning=static,
    )

    statistics = artifact_generator._record_metric_statistics(
        cast("Any", record),
        scaling=cast("Any", object()),
    )

    assert normalization_shapes == [(2, 4, 2, 2), (2, 4, 2, 2)]
    assert len(weight_calls) == 1
    assert statistics["cumulative"]["counts"] == [8, 8, 8, 8]
    assert statistics["endpoint"]["counts"] == [4, 4, 4, 4]


def test_protocol_identity_covers_complete_transient_evaluation_configuration() -> None:
    """Distinguish equal objective labels with different persisted protocol settings."""
    objective = {"id": "normalized_drying_group_macro_rmse"}
    first = artifact.evaluation_protocol_identity({"objective": objective, "metrics": [{"id": "metric"}], "rollout": {"origins": "early"}})
    second = artifact.evaluation_protocol_identity({"objective": objective, "metrics": [{"id": "metric"}], "rollout": {"origins": "all"}})
    assert first != second


def test_transient_artifact_plan_rejects_provisional_run(tmp_path: Any) -> None:
    """Require completed-run admission before projecting transient artifact roles."""
    with pytest.raises(ValueError, match="completed run"):
        artifact_generator.transient_artifact_plan_from_completed(
            {"is_completed": False},
            run_dir=tmp_path / "provisional-run",
        )


def test_shared_artifact_owner_dispatches_transient_before_steady_assumptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Route transient generation through its sequence owner before steady requests."""
    expected = pd.DataFrame({"sequence": [1]})
    observed: dict[str, Any] = {}
    monkeypatch.setattr(artifact_service, "_load_run_config", lambda _run_dir: {"task": "transient_drying"})
    monkeypatch.setattr(
        artifact_service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: SimpleNamespace(id="transient_drying"),
    )

    def transient_owner(**kwargs: Any) -> pd.DataFrame:
        observed.update(kwargs)
        return expected

    monkeypatch.setattr(artifact_service, "_run_or_load_transient_artifacts_locked", transient_owner)
    monkeypatch.setattr(
        artifact_service,
        "_build_artifact_request",
        lambda **_kwargs: pytest.fail("transient artifact dispatch reached the steady request owner"),
    )
    result = artifact_service._run_or_load_artifacts_locked(
        run_dir=tmp_path / "run",
        dataset_name="transient_dataset",
        split="eval",
        device_resolution=cast("Any", object()),
        dataset_root=tmp_path / "datasets",
        metadata_root=tmp_path / "metadata",
        rebuild=False,
    )
    assert result is expected
    assert observed["dataset_name"] == "transient_dataset"
    assert observed["split"] == "eval"


def _patch_publication_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict[str, Any],
    task_id: str,
    tracking_session: Any,
    events: list[str],
) -> None:
    """Replace the completed-run observer lifecycle with bounded local fakes."""
    summary = {
        "status": "completed",
        "completed_epoch": 3,
        "global_step": 12,
        "selected_epoch": 2,
        "selected_metrics": {},
        "terminal_epoch": 3,
        "terminal_metrics": {},
    }
    monkeypatch.setattr(artifact_service, "_load_run_config", lambda _run_dir: config)
    monkeypatch.setattr(
        artifact_service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: SimpleNamespace(id=task_id),
    )
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "initial_tracking_state",
        lambda _config: {},
    )
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "append_runtime_session",
        lambda *_args, **_kwargs: events.append("append_runtime_session"),
    )
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "read_run_summary",
        lambda _run_dir: summary,
    )
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "update_runtime_session",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        artifact_service.experiments.tracking,
        "persisted_wandb_identity",
        lambda _summary: ("fake-run-id", 3),
    )

    def initialize(*_args: Any, **_kwargs: Any) -> Any:
        events.append("initialize_wandb")
        return tracking_session

    monkeypatch.setattr(
        artifact_service.experiments.tracking,
        "initialize_wandb",
        initialize,
    )


@pytest.mark.parametrize(
    ("mode", "upload_enabled"),
    [("disabled", True), ("offline", False)],
)
def test_unconfigured_transient_publication_has_no_observer_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    mode: str,
    upload_enabled: bool,
) -> None:
    """Keep resolved W&B mode and upload policy as the only publication gates."""
    config = {
        "tracking": {
            "wandb": {
                "mode": mode,
                "upload": {"evaluation_artifacts": upload_enabled},
            }
        }
    }
    monkeypatch.setattr(artifact_service, "_load_run_config", lambda _run_dir: config)
    monkeypatch.setattr(
        artifact_service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: pytest.fail("unconfigured publication resolved its task"),
    )
    monkeypatch.setattr(
        artifact_service.experiments.tracking,
        "initialize_wandb",
        lambda *_args, **_kwargs: pytest.fail("unconfigured publication initialized W&B"),
    )
    artifact_service._upload_published_artifacts(
        plan=artifact_service.RunArtifactPlan(
            run_dir=tmp_path / "run",
            id_dataset_name="id_dataset",
            ood_dataset_name=None,
            lifecycle_status="completed",
            is_completed=True,
            scientific_run_name="run",
        ),
        device_resolution=cast("Any", object()),
        id_frame=pd.DataFrame(),
        ood_frame=None,
    )


@pytest.mark.parametrize("with_ood", [False, True])
def test_transient_artifact_publication_logs_only_validated_bounded_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    with_ood: bool,
) -> None:
    """Validate all roles before summary-only publication and exclude files/media."""
    events: list[str] = []
    payload: dict[str, bool | float | int | str] = {
        "evaluation/0/id/autonomous_full/full/cumulative/normalized_drying_group_macro_rmse": 0.25,
        "evaluation/0/id/autonomous_full/full/cumulative/contributing_case_count": 3,
        "evaluation/identity/backend": "canonical_hdf5",
    }
    frame_by_role = {"id": pd.DataFrame()}
    provenance_by_role: dict[str, dict[str, Any]] = {"id": {"role": "id", "identity": "a" * 64}}
    if with_ood:
        frame_by_role["ood"] = pd.DataFrame()
        provenance_by_role["ood"] = {"role": "ood", "identity": "b" * 64}
    for role, frame in frame_by_role.items():
        frame.attrs["artifact_provenance"] = provenance_by_role[role]

    def artifact_root(*, split: str, **_kwargs: Any) -> Any:
        return tmp_path / split

    def validate(*, artifact_root: Any, **_kwargs: Any) -> dict[str, Any]:
        role = "id" if artifact_root.name == "eval" else "ood"
        events.append(f"validate_{role}")
        return provenance_by_role[role]

    class FakeTransientSession:
        def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
            assert frames == frame_by_role
            events.append("reduce_summary")

        def wandb_publication_summary(self) -> dict[str, bool | float | int | str]:
            return payload

        def close(self) -> None:
            events.append("close_summary")

    logged: list[dict[str, bool | float | int | str]] = []
    tracking_session = SimpleNamespace(
        log_transient_evaluation_summary=lambda evidence: (events.append("log_summary"), logged.append(dict(evidence))),
        upload_files=lambda _files: pytest.fail("transient publication uploaded provenance"),
        upload_post_artifact=lambda **_kwargs: pytest.fail("transient publication uploaded media"),
        finish=lambda **_kwargs: events.append("finish_wandb"),
    )
    config = {
        "tracking": {
            "wandb": {
                "mode": "offline",
                "upload": {"evaluation_artifacts": True},
            }
        }
    }
    monkeypatch.setattr(artifact_service, "_artifact_save_root", artifact_root)
    monkeypatch.setattr(artifact_service, "validate_artifact_upload_source", validate)
    monkeypatch.setattr(transient_session, "TransientEvaluationSession", FakeTransientSession)
    _patch_publication_lifecycle(
        monkeypatch,
        config=config,
        task_id="transient_drying",
        tracking_session=tracking_session,
        events=events,
    )
    plan = artifact_service.RunArtifactPlan(
        run_dir=tmp_path / "run",
        id_dataset_name="id_dataset",
        ood_dataset_name="ood_dataset" if with_ood else None,
        lifecycle_status="completed",
        is_completed=True,
        scientific_run_name="run",
    )
    artifact_service._upload_published_artifacts(
        plan=plan,
        device_resolution=cast("Any", object()),
        id_frame=frame_by_role["id"],
        ood_frame=frame_by_role.get("ood"),
    )

    expected_roles = ["id", "ood"] if with_ood else ["id"]
    assert logged == [payload]
    assert all(events.index(f"validate_{role}") < events.index("initialize_wandb") for role in expected_roles)
    assert events.index("reduce_summary") < events.index("initialize_wandb") < events.index("log_summary")
    assert events[-1] == "finish_wandb"


def test_steady_artifact_publication_retains_provenance_and_curated_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Preserve the existing steady ID/OOD file and curated-report lifecycle."""
    events: list[str] = []
    file_uploads: list[dict[str, Any]] = []
    post_uploads: list[dict[str, Any]] = []
    tracking_session = SimpleNamespace(
        log_transient_evaluation_summary=lambda _evidence: pytest.fail("steady publication used transient summary"),
        upload_files=lambda files: file_uploads.append(dict(files)),
        upload_post_artifact=lambda **kwargs: post_uploads.append(dict(kwargs)),
        finish=lambda **_kwargs: events.append("finish_wandb"),
    )
    config = {
        "tracking": {
            "wandb": {
                "mode": "offline",
                "upload": {"evaluation_artifacts": True},
            }
        }
    }
    monkeypatch.setattr(
        artifact_service,
        "_artifact_save_root",
        lambda *, split, **_kwargs: tmp_path / split,
    )
    monkeypatch.setattr(
        artifact_service,
        "validate_artifact_upload_source",
        lambda **_kwargs: {"validated": True},
    )
    monkeypatch.setattr(evaluation_dataframe, "build_eval_df", lambda frame: frame)
    monkeypatch.setattr(
        curated,
        "render_curated_analysis",
        lambda **_kwargs: SimpleNamespace(media_files={}, tables={}),
    )
    _patch_publication_lifecycle(
        monkeypatch,
        config=config,
        task_id="steady_flow",
        tracking_session=tracking_session,
        events=events,
    )
    frame = pd.DataFrame({"value": [1.0]})
    artifact_service._upload_published_artifacts(
        plan=artifact_service.RunArtifactPlan(
            run_dir=tmp_path / "run",
            id_dataset_name="id_dataset",
            ood_dataset_name="ood_dataset",
            lifecycle_status="completed",
            is_completed=True,
            scientific_run_name="run",
        ),
        device_resolution=cast("Any", object()),
        id_frame=frame,
        ood_frame=frame,
    )

    assert len(file_uploads) == 2
    assert len(post_uploads) == 1
    assert post_uploads[0]["artifact_root"] == tmp_path / "eval"
    assert events[-1] == "finish_wandb"


def test_multi_package_role_rejects_duplicate_package_case_identity() -> None:
    """Prevent OOD package parts from colliding in sequence and timing identities."""
    identities = (
        {"dataset_id": "ood_a"},
        {"dataset_id": "ood_b"},
    )
    evidence = (
        {"case_ids": ["case"], "membership_digest": "a" * 64},
        {"case_ids": ["case"], "membership_digest": "b" * 64},
    )
    with pytest.raises(ValueError, match="duplicate package-case"):
        artifact_generator._role_plan(
            split="ood",
            dataset_role="ood",
            dataset_ids=("ood_a", "ood_b"),
            identities=identities,
            role_evidence=evidence,
        )


def test_scoped_role_selection_preserves_saved_order_and_rejects_unknown_cases() -> None:
    """Keep disposable case selection bounded to exact saved role membership."""
    role = artifact_generator.TransientArtifactRolePlan(
        split="ood",
        dataset_role="ood",
        dataset_name="combined-ood",
        source_dataset_ids=("ood-a", "ood-b"),
        source_identities=(
            {"dataset_id": "ood-a"},
            {"dataset_id": "ood-b"},
        ),
        case_ids_by_source=(("case-a", "case-b"), ("case-c",)),
        membership_digests=("a" * 64, "b" * 64),
    )

    selected = artifact_generator.select_transient_role_cases(
        role,
        ("case-c", "case-a"),
    )

    assert selected.case_ids_by_source == (("case-a",), ("case-c",))
    assert selected.membership_digests == role.membership_digests
    with pytest.raises(ValueError, match="absent from saved"):
        artifact_generator.select_transient_role_cases(role, ("missing",))


def test_default_geometric_horizons_are_exact_transition_contracts() -> None:
    """Derive field order from Dataset ownership and retain mandated horizons."""
    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    assert tuple(field.name for field in contract.dynamic_state) == artifact.STATE_ORDER
    assert tuple(field.name for field in contract.static_spatial_conditioning) == artifact.STATIC_ORDER
    assert tuple(field.name for field in contract.step_boundary_conditioning) == artifact.BOUNDARY_ORDER
    assert tuple(field.name for field in contract.scalar_conditioning) == artifact.SCALAR_ORDER
    assert artifact.FIXED_HORIZONS == (1, 2, 4, 8, 16, 32, 64, 128)
    process_policy = artifact_generator._process_diagnostic_policy()
    assert process_policy["bulk_moisture"]["available"] is True
    assert process_policy["stability_increment_growth_factor"] == 2.0
    assert process_policy["mass_balance"]["available"] is False


def test_prepared_evaluation_case_owns_and_guards_cpu_reference_states() -> None:
    """Keep CPU reference windows immutable after one strict prepared-case admission."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")
    context = _context()
    prepared = rollout.prepare_transient_evaluation_case(context, case)
    expected = prepared.reference_states.clone()

    case.reference_states[:] = -999.0
    window = rollout._prepared_request(
        prepared,
        origin=1,
        length=1,
    )

    assert torch.equal(window.state, expected[:, 1])
    prepared.reference_states.add_(1.0)
    with pytest.raises(RuntimeError, match="mutated after strict admission"):
        rollout.benchmark_transient_full_rollout(
            context,
            case,
            prepared_case=prepared,
            warmup_passes=0,
            repetitions=1,
        )


def test_rollout_modes_origins_horizons_conditioning_and_sequence_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Keep modes, origins, conditioning, availability, and persistence aligned."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")
    case = replace(
        case,
        metadata={
            **case.metadata,
            "has_exact_stop_state": True,
            "t_stop_exact": 3.5,
        },
    )
    autonomous_requests: list[inference.TransientPreparedRequest] = []

    def fake_step(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
    ) -> inference.TransientStepResult:
        state = request.state
        next_state = state.clone()
        next_state[:, 0] += 1.0
        next_state[:, 2:] -= 1.0
        return inference.TransientStepResult(
            next_state=next_state,
            scaled_delta=torch.zeros_like(state),
            timing=inference.TransientTiming(0.01, "cpu", "float32", "one_step", 1),
        )

    def fake_rollout(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
    ) -> inference.TransientRolloutResult:
        autonomous_requests.append(request)
        current = request.state.clone()
        length = int(request.boundary.shape[1])
        states: list[torch.Tensor] = []
        for _index in range(length):
            current = current.clone()
            current[:, 0] += 1.0
            current[:, 2:] -= 1.0
            states.append(current)
        return inference.TransientRolloutResult(
            states=torch.stack(states, dim=1),
            scaled_deltas=torch.zeros((1, length, 4, 2, 2), dtype=torch.float32),
            timing=inference.TransientTiming(0.02 * length, "cpu", "float32", "autonomous_rollout", length),
        )

    monkeypatch.setattr(inference, "predict_prepared_transient_step", fake_step)
    monkeypatch.setattr(
        inference,
        "rollout_prepared_transient_autonomous",
        fake_rollout,
    )
    benchmark = rollout.benchmark_transient_full_rollout(
        _context(),
        case,
        warmup_passes=2,
        repetitions=3,
    )
    assert len(autonomous_requests) == 6
    assert benchmark.warmup_passes == 2
    assert benchmark.warmed_model_seconds == pytest.approx((0.06, 0.06, 0.06))
    assert all(value > 0.0 for value in benchmark.warmed_end_to_end_seconds)
    autonomous_requests.clear()
    evaluated = rollout.evaluate_transient_case(
        _context(),
        case,
        identity=_identity(),
        reference_completion={
            "physical_duration_hours": 3.5,
            "time_to_target_hours": 3.5,
            "target_reached": True,
            "right_censored": False,
            "final_wet_fraction": 0.25,
            "target_wet_fraction_limit": 0.5,
            "physical_duration_availability": "available",
            "target_wet_fraction_limit_availability": "available",
        },
        target_wet_basis=0.05,
        target_fraction_limit=0.5,
        fixed_horizons=(1, 2, 4),
    )

    teacher = [record for record in evaluated.records if record.mode == "teacher_forced_one_step"]
    full = [record for record in evaluated.records if record.mode == "autonomous_full"]
    rolling = [record for record in evaluated.records if record.mode == "rolling_origin"]
    assert len(teacher) == 3
    assert len(full) == 1
    assert {(record.origin_index, record.requested_horizon) for record in rolling} == {
        (0, 1),
        (0, 2),
        (0, "full"),
        (1, 1),
        (1, 2),
        (1, "full"),
        (2, 1),
        (2, "full"),
    }
    assert len(evaluated.unavailable_horizons) == 4
    assert len(autonomous_requests) == 4
    assert [request.boundary.shape[1] for request in autonomous_requests] == [3, 3, 2, 1]
    assert torch.equal(
        autonomous_requests[-1].boundary[0, 0],
        torch.arange(9, dtype=torch.float32) + 2,
    )
    assert full[0].elapsed_physical_time == pytest.approx(3.0)
    assert full[0].target["reference_available"] is True
    assert full[0].target["reference_evidence_scope"] == "canonical_completed_case"
    assert full[0].target["reference_reached"] is True
    assert full[0].target["reference_time_to_target"] == pytest.approx(3.5)
    assert full[0].target["reference_final_time"] == pytest.approx(3.5)
    assert 3.5 not in full[0].physical_times
    assert full[0].target["predicted_censored"] is True
    assert full[0].target["predicted_final_time"] == pytest.approx(3.0)
    assert all(
        record.target["reference_evidence_scope"] == "unavailable_partial_interval" and record.target["reference_available"] is False
        for record in (*teacher, *rolling)
    )
    wrong_criterion = dict(full[0].target)
    wrong_criterion["criterion"] = "noncanonical"
    with pytest.raises(artifact.TransientSequenceArtifactError, match="canonical"):
        replace(full[0], target=wrong_criterion).validated()
    unsupported_time = dict(full[0].target)
    unsupported_time.update(
        {
            "predicted_available": True,
            "predicted_unavailable_reason": None,
            "predicted_reached": True,
            "predicted_censored": False,
            "predicted_time_to_target": 1.5,
            "predicted_final_gap": 0.0,
        }
    )
    with pytest.raises(artifact.TransientSequenceArtifactError, match="exact physical time"):
        replace(full[0], target=unsupported_time).validated()
    invalid_times = full[0].physical_times.copy()
    invalid_times[1] = invalid_times[0]
    with pytest.raises(artifact.TransientSequenceArtifactError, match="strictly increasing"):
        replace(full[0], physical_times=invalid_times).validated()

    provenance = {
        "run": {
            "name": "run",
            "checkpoint_sha256": "b" * 64,
            "parent_experiment": {
                "kind": "legacy",
                "parent_available": False,
                "reason": "no_validated_parent_experiment_for_exact_child_path",
                "child_source_repository": {"commit": None, "dirty": None},
            },
        },
        "dataset": {"name": "transient_dataset", "role": "id", "identity": {"index_digest": "a" * 64}},
        "evaluation": {"config_identity": "d" * 64},
        "runtime": {"device": "cpu", "precision": "float32"},
        "lineage": {"strategy": "rollout"},
    }
    with pytest.raises(ValueError, match="unique available records"):
        artifact.write_transient_sequence_artifact(
            tmp_path / "duplicate",
            dataset_name="transient_dataset",
            dataset_role="id",
            records=(full[0], full[0]),
            provenance=provenance,
        )

    stored = artifact.write_transient_sequence_artifact(
        tmp_path / "artifact",
        dataset_name="transient_dataset",
        dataset_role="id",
        records=evaluated.records,
        unavailable_horizons=evaluated.unavailable_horizons,
        provenance=provenance,
    )
    assert stored.identity_sha256
    assert len(stored.summaries) == len(evaluated.records)
    assert stored.unavailable_horizons == tuple(item.as_dict() for item in evaluated.unavailable_horizons)
    assert stored.summaries[0].identity["checkpoint_identity"] == {"sha256": "b" * 64}
    stored_full = next(record for record in stored.summaries if record.mode == "autonomous_full")
    assert stored_full.target["reference_time_to_target"] == pytest.approx(3.5)
    identity_before_telemetry = stored.identity_sha256
    recorder = artifact_performance.ArtifactPerformanceRecorder(
        device="cpu",
        dtype="float32",
    )
    with recorder.stage("inference"):
        recorder.increment("case_count")
        recorder.increment("forward_call_count", 3)
    artifact.publish_transient_operational_performance(
        tmp_path / "artifact",
        recorder.snapshot(),
    )
    with_telemetry = artifact.load_transient_sequence_artifact_index(tmp_path / "artifact")
    assert with_telemetry.identity_sha256 == identity_before_telemetry
    assert with_telemetry.provenance["runtime"]["operational_performance"]["counts"] == {
        "case_count": 1,
        "forward_call_count": 3,
    }
    assert len(tuple((tmp_path / "artifact" / "npz").glob("*.npz"))) == 1
    persisted_rows = pd.read_parquet(tmp_path / "artifact" / "transient_dataset.parquet")
    assert len(set(persisted_rows["payload_path"])) == 1
    for origin in (0, 1, 2):
        rolling = persisted_rows[(persisted_rows["mode"] == "rolling_origin") & (persisted_rows["origin_index"] == origin)]
        assert len(set(rolling["chain_id"])) == 1

    artifact_root = tmp_path / "artifact"
    parquet_path = artifact_root / "transient_dataset.parquet"
    marker_path = artifact_root / artifact_contracts.ARTIFACT_PROVENANCE_FILENAME
    original_parquet = parquet_path.read_bytes()
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    mismatched_rows = persisted_rows.copy()
    mismatched_rows.loc[0, "case_id"] = "different-case"
    mismatched_rows.to_parquet(parquet_path, index=False)
    marker["outputs"] = artifact_contracts.artifact_output_manifest(artifact_root)
    common.serialization.atomic_write_json(marker_path, marker)
    with pytest.raises(artifact.TransientSequenceArtifactError, match="identity contradicts"):
        artifact.load_transient_sequence_artifact_index(artifact_root)
    parquet_path.write_bytes(original_parquet)
    marker["outputs"] = artifact_contracts.artifact_output_manifest(artifact_root)
    common.serialization.atomic_write_json(marker_path, marker)

    calls: list[str] = []
    payload_loads: list[str] = []
    original_loader = artifact._record_from_row
    original_np_load = artifact.np.load

    def counted_np_load(*args: Any, **kwargs: Any) -> Any:
        payload_loads.append(str(args[0]))
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(artifact.np, "load", counted_np_load)

    def counted_loader(root: Any, row: Any, payload: Any = None) -> artifact.TransientSequenceRecord:
        calls.append(str(row["record_id"]))
        return original_loader(root, row, payload)

    monkeypatch.setattr(artifact, "_record_from_row", counted_loader)
    indexed = artifact.load_transient_sequence_artifact_index(tmp_path / "artifact")
    payload_path = tmp_path / "artifact" / persisted_rows["payload_path"].iloc[0]
    assert indexed.case_ids == ("case",)
    assert indexed.cache_size == 0
    assert indexed.cache_limit == 1
    assert calls == []
    artifact.validate_transient_sequence_payload_inventory(indexed)
    assert payload_loads == [str(payload_path)]
    assert indexed.cache_size == 0
    payload_loads.clear()
    selected_case = indexed.records(case_id="case")
    assert len(selected_case) == len(evaluated.records)
    assert len(calls) == len(evaluated.records)
    assert payload_loads == [str(payload_path)]
    assert indexed.cache_size == 1

    with original_np_load(payload_path, allow_pickle=False) as loaded_payload:
        corrupted_payload = {name: loaded_payload[name] for name in loaded_payload.files}
    np.savez_compressed(payload_path, **corrupted_payload, unexpected=np.asarray([1]))
    with pytest.raises(artifact.TransientSequenceArtifactError, match="output manifest"):
        artifact.load_transient_sequence_artifact_index(artifact_root)
    with pytest.raises(artifact.TransientSequenceArtifactError, match="field inventory"):
        artifact.validate_transient_sequence_payload_inventory(indexed)

    with pytest.raises(ValueError, match="sequence inventory"):
        artifact_generator._validate_sequence_inventory(
            tuple(record for record in stored.summaries if record.mode != "autonomous_full"),
            stored.unavailable_horizons,
            expected_cases={"case"},
            dataset_role="id",
        )


def _cache_test_record(case_id: str, *, mode: artifact.EvaluationMode) -> artifact.TransientSequenceRecord:
    """Build one minimal strict record for case-payload cache coverage."""
    identity = _identity()
    identity["case_id"] = case_id
    identity["simulation_identity"] = {"simulation_case_id": case_id, "source_batch_id": "batch"}
    full_autonomous = mode == "autonomous_full"
    target = {
        "criterion": artifact.TARGET_CRITERION,
        "limit": 0.5,
        "reference_evidence_scope": "canonical_completed_case" if full_autonomous else "unavailable_partial_interval",
        "predicted_evidence_scope": "regular_sequence_grid",
        "reference_available": full_autonomous,
        "predicted_available": True,
        "reference_unavailable_reason": None if full_autonomous else "partial sequence",
        "predicted_unavailable_reason": None,
        "reference_reached": False,
        "predicted_reached": False,
        "reference_censored": full_autonomous,
        "predicted_censored": True,
        "reference_time_to_target": None,
        "predicted_time_to_target": None,
        "reference_final_gap": 0.1 if full_autonomous else None,
        "predicted_final_gap": 0.1,
        "reference_final_time": 1.0 if full_autonomous else None,
        "predicted_final_time": 1.0,
    }
    return artifact.TransientSequenceRecord(
        mode=mode,
        case_id=case_id,
        dataset_role="id",
        origin_index=0,
        requested_horizon="full" if full_autonomous else 1,
        available_horizon=1,
        trajectory_length=2,
        physical_times=np.asarray([0.0, 1.0], dtype=np.float64),
        transition_indices=np.asarray([0], dtype=np.int64),
        reference_states=np.zeros((2, len(artifact.STATE_ORDER), 1, 1), dtype=np.float32),
        predicted_states=np.ones((2, len(artifact.STATE_ORDER), 1, 1), dtype=np.float32),
        reference_increments=None,
        predicted_increments=None,
        spatial_mask=np.ones((1, 1), dtype=bool),
        temporal_mask=np.ones(2, dtype=bool),
        static_conditioning=np.zeros((len(artifact.STATIC_ORDER), 1, 1), dtype=np.float32),
        boundary_conditioning=np.zeros((1, len(artifact.BOUNDARY_ORDER)), dtype=np.float32),
        scalar_conditioning=np.zeros(len(artifact.SCALAR_ORDER), dtype=np.float32),
        identity=identity,
        target=target,
        timing={},
        exclusion={"excluded": False, "reason": None},
    ).validated()


def _cache_test_provenance() -> dict[str, Any]:
    """Return the minimum strict provenance required for a cache fixture."""
    return {
        "run": {
            "name": "run",
            "checkpoint_sha256": "b" * 64,
            "parent_experiment": {
                "kind": "legacy",
                "parent_available": False,
                "reason": "no_validated_parent_experiment_for_exact_child_path",
                "child_source_repository": {"commit": None, "dirty": None},
            },
        },
        "dataset": {"name": "transient_dataset", "role": "id", "identity": {"index_digest": "a" * 64}},
        "evaluation": {"config_identity": "d" * 64},
        "runtime": {"device": "cpu", "precision": "float32"},
        "lineage": {"strategy": "rollout"},
    }


def test_streaming_stager_writes_cases_before_marker_and_retains_only_compact_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Persist each case once while keeping private staging incomplete until finalization."""
    root = tmp_path / "streamed"
    observed_hashes: list[Path] = []
    original_sha256 = artifact.common.serialization.file_sha256

    def counted_sha256(value: Path) -> str:
        observed_hashes.append(value)
        return original_sha256(value)

    monkeypatch.setattr(
        artifact.common.serialization,
        "file_sha256",
        counted_sha256,
    )
    stager = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
    )
    first = (
        _cache_test_record("stream-a", mode="autonomous_full"),
        _cache_test_record("stream-a", mode="teacher_forced_one_step"),
    )
    second = (
        _cache_test_record("stream-b", mode="autonomous_full"),
        _cache_test_record("stream-b", mode="teacher_forced_one_step"),
    )

    stager.write_case(first)

    marker = artifact_contracts.artifact_provenance_path(root)
    first_payload = root / artifact._case_payload_path("stream-a")
    assert first_payload.is_file()
    assert not marker.exists()
    assert len(stager._rows) == 2
    assert not any(isinstance(value, np.ndarray) for row in stager._rows for value in row.values())

    stager.write_case(second)
    index = stager.finalize(provenance=_cache_test_provenance())

    assert marker.is_file()
    assert index.case_ids == ("stream-a", "stream-b")
    assert observed_hashes.count(first_payload) == 1
    assert len([path for path in observed_hashes if path.suffix == ".npz"]) == 2
    assert len([path for path in observed_hashes if path.suffix == ".parquet"]) == 1
    for payload in (root / "npz").glob("*.npz"):
        with zipfile.ZipFile(payload) as archive:
            assert {entry.compress_type for entry in archive.infolist()} == {zipfile.ZIP_STORED}


def test_scoped_index_validator_requires_exact_noncanonical_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject malformed scoped evidence before admitting selected role membership."""
    saved_role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=(("case",),),
        membership_digests=("a" * 64,),
    )
    malformed = _cache_test_provenance()
    malformed["evaluation"]["scope"] = {"kind": "full"}
    malformed_root = tmp_path / "malformed-scoped"
    artifact.write_transient_sequence_artifact(
        malformed_root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(_cache_test_record("case", mode="autonomous_full"),),
        provenance=malformed,
    )

    with pytest.raises(ValueError, match="exact selected-case scope"):
        artifact_generator.validate_scoped_transient_role_artifact_index(
            malformed_root,
            completed={},
            saved_role=saved_role,
        )

    scoped = _cache_test_provenance()
    scoped["evaluation"]["scope"] = {
        "kind": "selected_cases",
        "canonical_publication_eligible": False,
        "selected_case_ids": ["case"],
        "saved_case_ids_by_source": [["case"]],
    }
    scoped_root = tmp_path / "valid-scoped"
    artifact.write_transient_sequence_artifact(
        scoped_root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(_cache_test_record("case", mode="autonomous_full"),),
        provenance=scoped,
    )
    monkeypatch.setattr(
        artifact_generator,
        "_validate_transient_role_evidence",
        lambda *_args, **_kwargs: None,
    )

    admitted = artifact_generator.validate_scoped_transient_role_artifact_index(
        scoped_root,
        completed={},
        saved_role=saved_role,
    )

    assert admitted.case_ids == ("case",)


def test_transient_reader_remains_backward_compatible_with_compressed_npz(
    tmp_path: Any,
) -> None:
    """Admit prior deflated NPZ bundles through the unchanged strict reader."""
    root = tmp_path / "compressed"
    artifact.write_transient_sequence_artifact(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(
            _cache_test_record("legacy-compressed", mode="autonomous_full"),
            _cache_test_record(
                "legacy-compressed",
                mode="teacher_forced_one_step",
            ),
        ),
        provenance=_cache_test_provenance(),
    )
    payload_path = next((root / "npz").glob("*.npz"))
    with np.load(payload_path, allow_pickle=False) as loaded:
        payload = {name: loaded[name] for name in loaded.files}
    temporary = payload_path.with_suffix(".compressed.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(payload_path)
    payload_digest = artifact.common.serialization.file_sha256(payload_path)
    parquet_path = root / "transient_dataset.parquet"
    frame = pd.read_parquet(parquet_path)
    frame["payload_sha256"] = payload_digest
    frame.to_parquet(parquet_path, index=False)
    marker = artifact_contracts.artifact_provenance_path(root)
    provenance = json.loads(marker.read_text())
    provenance["outputs"] = artifact_contracts.artifact_output_manifest(root)
    artifact.common.serialization.atomic_write_json(marker, provenance)

    admitted = artifact.load_transient_sequence_artifact_index(root)

    assert admitted.record(admitted.record_ids[0]).case_id == "legacy-compressed"
    with zipfile.ZipFile(payload_path) as archive:
        assert zipfile.ZIP_DEFLATED in {entry.compress_type for entry in archive.infolist()}


def test_transient_artifact_index_lazily_opens_case_payloads_and_closes_lru_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Open one selected case archive lazily, reuse it, and close it on eviction."""
    first_case = "cache-case-a"
    second_case = "cache-case-b"
    artifact_root = tmp_path / "artifact"
    artifact.write_transient_sequence_artifact(
        artifact_root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(
            _cache_test_record(first_case, mode="autonomous_full"),
            _cache_test_record(first_case, mode="teacher_forced_one_step"),
            _cache_test_record(second_case, mode="autonomous_full"),
            _cache_test_record(second_case, mode="teacher_forced_one_step"),
        ),
        provenance=_cache_test_provenance(),
    )
    original_sha256 = artifact.common.serialization.file_sha256
    digest_paths: list[Path] = []

    def counted_sha256(value: Path) -> str:
        digest_paths.append(value)
        return original_sha256(value)

    monkeypatch.setattr(artifact.common.serialization, "file_sha256", counted_sha256)
    index = artifact.load_transient_sequence_artifact_index(artifact_root, cache_limit=1)
    assert digest_paths == []

    original_np_load = artifact.np.load
    opened: list[Path] = []
    closed: list[Path] = []

    class TrackedArchive:
        def __init__(self, path: Path, archive: Any) -> None:
            self.path = path
            self.archive = archive
            self.files = archive.files

        def __getitem__(self, key: str) -> Any:
            return self.archive[key]

        def close(self) -> None:
            closed.append(self.path)
            self.archive.close()

    def counted_np_load(value: Path, *args: Any, **kwargs: Any) -> Any:
        archive = original_np_load(value, *args, **kwargs)
        path = Path(value)
        opened.append(path)
        return TrackedArchive(path, archive)

    monkeypatch.setattr(artifact.np, "load", counted_np_load)
    first_records = tuple(summary.record_id for summary in index.summaries if summary.case_id == first_case)
    second_records = tuple(summary.record_id for summary in index.summaries if summary.case_id == second_case)
    first_path = artifact_root / next(summary.payload_path for summary in index.summaries if summary.case_id == first_case)
    second_path = artifact_root / next(summary.payload_path for summary in index.summaries if summary.case_id == second_case)

    index.record(first_records[0])
    assert digest_paths == [first_path]
    assert opened == [first_path]
    index.record(first_records[1])
    assert opened == [first_path]
    index.record(second_records[0])
    assert opened == [first_path, second_path]
    assert closed == [first_path]
    index.record(first_records[0])
    assert opened == [first_path, second_path, first_path]
    assert closed == [first_path, second_path]
    index.close()
    assert closed == [first_path, second_path, first_path]

    first_path.write_bytes(first_path.read_bytes() + b"corrupt")
    with pytest.raises(artifact.TransientSequenceArtifactError, match="output manifest"):
        artifact.load_transient_sequence_artifact_index(artifact_root)


def test_separate_artifact_indexes_retain_case_local_caches_across_role_switches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Open only each owning payload and retain both role-local LRU entries."""

    def role_record(
        case_id: str,
        *,
        role: artifact.DatasetRole,
        material: str,
    ) -> artifact.TransientSequenceRecord:
        record = _cache_test_record(case_id, mode="autonomous_full")
        identity = dict(record.identity)
        identity["dataset_role"] = role
        identity["material_family"] = material
        identity["dataset_identity"] = {
            "dataset_id": f"dataset_{material}",
            "index_digest": material.ljust(64, "a")[:64],
        }
        return replace(
            record,
            dataset_role=role,
            identity=identity,
        ).validated()

    id_root = tmp_path / "id"
    ood_root = tmp_path / "ood"
    id_provenance = _cache_test_provenance()
    id_provenance["dataset"] = {
        "name": "dataset_lentil_chickpea",
        "role": "id",
        "identity": {"index_digest": "a" * 64},
    }
    ood_provenance = _cache_test_provenance()
    ood_provenance["dataset"] = {
        "name": "dataset_kidney_bean",
        "role": "ood",
        "identity": {"index_digest": "f" * 64},
    }
    artifact.write_transient_sequence_artifact(
        id_root,
        dataset_name="dataset_lentil_chickpea",
        dataset_role="id",
        records=(
            role_record(
                "transient_drying__chickpea__natural__case_0051",
                role="id",
                material="chickpea",
            ),
        ),
        provenance=id_provenance,
    )
    artifact.write_transient_sequence_artifact(
        ood_root,
        dataset_name="dataset_kidney_bean",
        dataset_role="ood",
        records=(
            role_record(
                "transient_drying__kidney_bean__natural__case_0004",
                role="ood",
                material="kidney_bean",
            ),
        ),
        provenance=ood_provenance,
    )
    id_index = artifact.load_transient_sequence_artifact_index(id_root)
    ood_index = artifact.load_transient_sequence_artifact_index(ood_root)
    original_np_load = artifact.np.load
    opened: list[Path] = []

    def counted_np_load(value: Path, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(value))
        return original_np_load(value, *args, **kwargs)

    monkeypatch.setattr(artifact.np, "load", counted_np_load)
    id_index.record(id_index.record_ids[0])
    id_index.record(id_index.record_ids[0])
    ood_index.record(ood_index.record_ids[0])
    id_index.record(id_index.record_ids[0])

    assert len(opened) == 2
    assert opened[0].is_relative_to(id_root)
    assert opened[1].is_relative_to(ood_root)
    assert id_index.cache_size == 1
    assert ood_index.cache_size == 1
    id_index.close()
    ood_index.close()


def test_parent_experiment_provenance_requires_exact_grouped_or_legacy_evidence() -> None:
    """Reject parent/source provenance that could misidentify one transient child run."""
    legacy = artifact._validate_parent_experiment_evidence(
        {
            "kind": "legacy",
            "parent_available": False,
            "reason": "no_validated_parent_experiment_for_exact_child_path",
            "child_source_repository": {"commit": "a" * 40, "dirty": False},
        }
    )
    assert legacy["kind"] == "legacy"
    grouped = artifact._validate_parent_experiment_evidence(
        {
            "kind": "grouped",
            "parent_available": True,
            "parent_label": "transient_parent",
            "parent_identity_sha256": "b" * 64,
            "run_revision": 2,
            "source_repository": {"commit": "c" * 40, "dirty": False},
            "child_source_repository": {"commit": "c" * 40, "dirty": True},
        }
    )
    assert grouped["parent_identity_sha256"] == "b" * 64
    with pytest.raises(artifact.TransientSequenceArtifactError, match="incomplete"):
        artifact._validate_parent_experiment_evidence(
            {
                "kind": "grouped",
                "parent_available": True,
                "parent_label": "transient_parent",
                "parent_identity_sha256": "b" * 64,
                "run_revision": 2,
                "source_repository": {"commit": "c" * 40, "dirty": False},
            }
        )


def test_cuda_model_timing_uses_event_abstraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Protect CUDA-event model timing without requiring a physical CUDA device."""

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True

        def record(self, _stream: object) -> None:
            return None

        def synchronize(self) -> None:
            return None

        def elapsed_time(self, _other: object) -> float:
            return 12.5

    def fake_predict(
        _model: nn.Module,
        step_input: torch.Tensor,
        *,
        model_kind: str,
        hidden: object | None,
        model_call: Any,
    ) -> tuple[torch.Tensor, object | None]:
        assert model_kind == "rno"
        return cast("torch.Tensor", model_call(lambda: torch.zeros_like(step_input[:, :4]))), hidden

    cuda_context = inference.TransientInferenceContext(
        model=nn.Identity(),
        tensorizer=cast("Any", object()),
        scaling=cast("Any", SimpleNamespace()),
        device=torch.device("cuda:0"),
        model_kind="rno",
        precision="float32",
    )
    monkeypatch.setattr(inference.rollout, "predict_step", fake_predict)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    _prediction, _hidden, seconds = inference._timed_predict(  # pyright: ignore[reportPrivateUsage]
        cuda_context,
        torch.zeros((1, 4, 2, 2), dtype=torch.float32),
        hidden=None,
    )
    assert seconds == pytest.approx(0.0125)


def test_generator_timing_adapter_preserves_stable_runtime_and_backend_evidence() -> None:
    """Map public rollout clocks and stable Generation timing without private parsing."""
    benchmark = rollout.TransientRolloutBenchmark(
        case_id="case",
        warmup_passes=1,
        repetitions=3,
        model_clock="time.perf_counter",
        wall_clock="time.perf_counter",
        model_calls_per_repetition=3,
        cold_model_seconds=9.0,
        cold_end_to_end_seconds=10.0,
        warmed_model_seconds=(1.0, 2.0, 3.0),
        warmed_end_to_end_seconds=(2.0, 3.0, 4.0),
    )
    adapted = artifact_generator._timing_case(
        case_id="case",
        benchmark=benchmark,
        generation_timing={
            "comsol_process_seconds": 30.0,
            "generation_compute_end_to_end_seconds": 60.0,
            "component_timing_availability": {
                "transient_drying_solver_seconds": "not_persisted",
            },
        },
        dataset_backend="pt_shards",
        pt_identity={"receipt_digest": "a" * 64},
        runtime_metadata={
            "resolved_device": "cpu",
            "processor": "test-cpu",
            "cuda_device_name": None,
            "python_version": "test",
            "pytorch_version": "test",
        },
    )
    assert adapted.repetitions["drying_no_rollout_model_seconds"] == (1.0, 2.0, 3.0)
    assert adapted.cold_timings["drying_no_rollout_model_seconds"] == 9.0
    assert adapted.repetitions["comsol_process_seconds"] == (30.0,)
    assert adapted.unavailable_reasons["airflow_no_model_seconds"] == "compatible_airflow_model_not_selected"
    assert adapted.dataset_backend == "pt_shards"
    assert adapted.pt_payload_identity is not None


def test_artifact_rejects_unsupported_model_alias() -> None:
    """Reject hidden UNO-RNO aliases at the sequence admission boundary."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")
    identity = _identity(model_kind="uno_rno")
    with pytest.raises(ValueError, match="model kind"):
        artifact.TransientSequenceRecord(
            mode="teacher_forced_one_step",
            case_id="case",
            dataset_role="id",
            origin_index=0,
            requested_horizon=1,
            available_horizon=1,
            trajectory_length=4,
            physical_times=case.physical_times[:2],
            transition_indices=np.asarray([0]),
            reference_states=case.reference_states[:2],
            predicted_states=case.reference_states[:2],
            reference_increments=np.diff(case.reference_states[:2], axis=0),
            predicted_increments=np.diff(case.reference_states[:2], axis=0),
            spatial_mask=case.spatial_mask,
            temporal_mask=np.ones(2, dtype=bool),
            static_conditioning=case.static_conditioning,
            boundary_conditioning=case.boundary_conditioning[:1],
            scalar_conditioning=case.scalar_conditioning,
            identity=identity,
            target={
                "criterion": artifact.TARGET_CRITERION,
                "limit": 0.5,
                "reference_evidence_scope": "unavailable_partial_interval",
                "predicted_evidence_scope": "regular_sequence_grid",
                "reference_available": False,
                "predicted_available": True,
                "reference_unavailable_reason": "canonical_completed_case_target_unavailable_for_partial_interval",
                "predicted_unavailable_reason": None,
                "reference_reached": False,
                "predicted_reached": False,
                "reference_time_to_target": None,
                "predicted_time_to_target": None,
                "reference_censored": False,
                "predicted_censored": True,
                "reference_final_gap": None,
                "predicted_final_gap": 0.1,
                "reference_final_time": None,
                "predicted_final_time": 1.0,
            },
            timing={},
            exclusion={"excluded": False, "reason": None},
        ).validated()


def test_scoped_transient_artifact_rejects_output_below_run_bundle(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep disposable debug payloads completely outside immutable run bundles."""
    run_dir = tmp_path / "completed-run"
    run_dir.mkdir()
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=(("case",),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="scientific-run",
        id_role=role,
        ood_role=None,
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "load_transient_artifact_plan",
        lambda _path: plan,
    )

    with pytest.raises(ValueError, match="outside the run"):
        artifact_service.build_scoped_transient_artifact(
            run_dir=run_dir,
            dataset_root=tmp_path / "02_datasets" / "packages",
            output_root=run_dir / "debug-artifact",
            split="id",
            one_case=True,
        )
