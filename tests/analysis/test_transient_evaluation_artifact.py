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
from src.analysis.evaluation import evaluation_transient_validity as transient_validity
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


def _spatial_identity(static: np.ndarray | None = None) -> dict[str, Any]:
    """Return exact 2x2 source-grid identity for compact sequence fixtures."""
    conditioning = _items()[0]["static"].numpy() if static is None else np.asarray(static, dtype=np.float32)
    representation = datasets.contracts.transient.resolve_spatial_representation((2, 2), 1)
    return artifact.build_transient_spatial_identity(
        representation,
        reference_states=np.zeros((2, len(artifact.STATE_ORDER), 2, 2), dtype=np.float32),
        static_conditioning=conditioning,
        spatial_mask=np.ones((2, 2), dtype=bool),
    )


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
        "spatial_representation": _spatial_identity(),
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


def _prediction_fields(
    *,
    predicted_states: np.ndarray,
    predicted_increments: np.ndarray,
    physical_times: np.ndarray,
    mode: artifact.EvaluationMode,
    origin_index: int,
) -> dict[str, Any]:
    """Build one internally consistent compact prediction diagnostic payload."""
    scaled_outputs = np.ascontiguousarray(predicted_increments, dtype=np.float32)
    availability = np.ones(predicted_increments.shape[0], dtype=bool)
    predicted_next = np.ascontiguousarray(predicted_states[1:], dtype=np.float32)
    return {
        "predicted_increments": scaled_outputs,
        "scaled_model_outputs": scaled_outputs,
        "prediction_available": availability,
        "prediction_nonfinite_mask": ~np.isfinite(predicted_next),
        "prediction_physical_invalid_mask": (transient_validity.prediction_physical_invalid_mask(predicted_next)),
        "prediction_validity": transient_validity.build_prediction_validity(
            scaled_model_outputs=scaled_outputs,
            decoded_physical_increments=scaled_outputs,
            reconstructed_states=predicted_next,
            prediction_available=availability,
            physical_times=physical_times,
            mode=mode,
            origin_index=origin_index,
        ),
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
        training_spatial_shape=(2, 2),
        evaluation_spatial_shape=(2, 2),
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
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
        spatial_representation=_spatial_identity(),
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
        available_horizon=2,
        prediction_available=np.ones(2, dtype=bool),
    )

    statistics = artifact_generator._record_metric_statistics(
        cast("Any", record),
        scaling=cast("Any", object()),
    )

    cumulative_statistics = cast("dict[str, Any]", statistics["cumulative"]["statistics"])
    endpoint_statistics = cast("dict[str, Any]", statistics["endpoint"]["statistics"])
    assert normalization_shapes == [(2, 4, 2, 2), (2, 4, 2, 2)]
    assert len(weight_calls) == 1
    assert statistics["cumulative"]["available"] is True
    assert statistics["endpoint"]["available"] is True
    assert cumulative_statistics["counts"] == [8, 8, 8, 8]
    assert endpoint_statistics["counts"] == [4, 4, 4, 4]


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


@pytest.mark.parametrize(
    ("training_stage", "comparison_arm"),
    [("a", "a0"), ("a", "a_plus"), ("b", "b")],
)
def test_artifact_evaluation_grid_resolution_defaults_original_and_honors_explicit_two(
    monkeypatch: pytest.MonkeyPatch,
    training_stage: str,
    comparison_arm: str,
) -> None:
    """Reconcile stride-two Training evidence with original-grid default artifact inference."""
    scaling = SimpleNamespace(
        dataset_identity={
            "spatial_stride": 2,
            "canonical_spatial_shape": [251, 401],
            "effective_spatial_shape": [126, 201],
        },
        spatial_shape=(126, 201),
    )
    monkeypatch.setattr(
        artifact_generator.TransientScalingArtifact,
        "from_state_dict",
        lambda _state: scaling,
    )
    completed = {
        "is_completed": True,
        "config": {
            "task": "transient_drying",
            "data": {"spatial_stride": 2},
            "training": {"stage": training_stage, "comparison_arm": comparison_arm},
        },
        "split_indices": {
            "spatial_view": {
                "spatial_stride": 2,
                "canonical_ny": 251,
                "canonical_nx": 401,
                "effective_ny": 126,
                "effective_nx": 201,
            }
        },
        "normalizer_state": {},
    }

    training, default_evaluation = artifact_generator.resolve_transient_artifact_spatial_representations(completed)
    _, sampled_evaluation = artifact_generator.resolve_transient_artifact_spatial_representations(
        completed,
        evaluation_spatial_stride=2,
    )

    assert training.spatial_stride == 2
    assert training.represented_shape == (126, 201)
    assert default_evaluation.spatial_stride == 1
    assert default_evaluation.represented_shape == (251, 401)
    assert sampled_evaluation.spatial_stride == 2
    assert sampled_evaluation.represented_shape == (126, 201)


def test_evaluation_grid_variants_have_disjoint_rebuild_and_lock_targets(tmp_path: Path) -> None:
    """Keep canonical and lower-grid cache publication independently recoverable."""
    run_dir = tmp_path / "run"
    canonical = artifact_service._artifact_save_root(
        run_dir=run_dir,
        dataset_name="dataset",
        split="eval",
    )
    sampled = artifact_service._artifact_save_root(
        run_dir=run_dir,
        dataset_name="dataset",
        split="eval",
        evaluation_spatial_stride=2,
    )
    sampled_ood = artifact_service._artifact_save_root(
        run_dir=run_dir,
        dataset_name="dataset",
        split="ood",
        evaluation_spatial_stride=2,
    )

    assert canonical == run_dir / "analysis" / "id"
    assert sampled == run_dir / "analysis" / "grid_s2" / "id"
    assert sampled_ood == run_dir / "analysis" / "grid_s2" / "ood" / "dataset"
    assert artifact_service._artifact_lock_path(run_dir=run_dir, save_root=canonical) != artifact_service._artifact_lock_path(
        run_dir=run_dir,
        save_root=sampled,
    )
    canonical.mkdir(parents=True)
    sampled.mkdir(parents=True)
    (canonical / "canonical.marker").write_text("canonical", encoding="utf-8")
    (sampled / "sampled.marker").write_text("sampled", encoding="utf-8")

    artifact_service._rebuild_artifact_target_locked(
        run_dir=run_dir,
        save_root=canonical,
    )

    assert not canonical.exists()
    assert (sampled / "sampled.marker").read_text(encoding="utf-8") == "sampled"


def test_transient_upload_admission_resolves_exact_grid_variant(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Revalidate a noncanonical upload target through its transient role and stride."""
    run_dir = tmp_path / "run"
    artifact_root = run_dir / "analysis" / "grid_s2" / "id"
    artifact_root.mkdir(parents=True)
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="dataset",
        source_dataset_ids=("dataset",),
        source_identities=({"dataset_id": "dataset"},),
        case_ids_by_source=(("case",),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="run",
        id_role=role,
        ood_role=None,
    )
    completed = {"config": {"task": "transient_drying"}}
    observed: dict[str, object] = {}

    def validate(_root: Path, **kwargs: object) -> SimpleNamespace:
        observed.update(kwargs)
        return SimpleNamespace(provenance={"validated": True})

    monkeypatch.setattr(artifact_service.experiments.run, "validate_completed_run", lambda _run_dir: completed)
    monkeypatch.setattr(
        artifact_service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: SimpleNamespace(id="transient_drying"),
    )
    monkeypatch.setattr(artifact_generator, "transient_artifact_plan_from_completed", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(artifact_generator, "validate_transient_role_artifact", validate)

    provenance = artifact_service.validate_artifact_upload_source(
        run_dir=run_dir,
        artifact_root=artifact_root,
    )

    assert provenance == {"validated": True}
    assert observed["completed"] is completed
    assert observed["role"] is role
    assert observed["evaluation_spatial_stride"] == 2


def test_grid_identity_distinguishes_equal_shapes_with_different_indices_or_coordinates() -> None:
    """Reject shape-only grid equality while leaving canonical scientific arrays unchanged."""
    state = np.zeros((2, len(artifact.STATE_ORDER), 3, 3), dtype=np.float32)
    static = np.zeros((len(artifact.STATIC_ORDER), 3, 3), dtype=np.float32)
    x_coordinates, y_coordinates = np.meshgrid(
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
    )
    static[artifact.STATIC_ORDER.index("x")] = x_coordinates
    static[artifact.STATIC_ORDER.index("y")] = y_coordinates
    mask = np.ones((3, 3), dtype=bool)
    state_before = state.copy()
    static_before = static.copy()
    sampled_source = datasets.contracts.transient.resolve_spatial_representation((5, 5), 2)
    compact_source = datasets.contracts.transient.resolve_spatial_representation((3, 3), 1)

    sampled_identity = artifact.build_transient_spatial_identity(
        sampled_source,
        reference_states=state,
        static_conditioning=static,
        spatial_mask=mask,
    )
    different_indices = artifact.build_transient_spatial_identity(
        compact_source,
        reference_states=state,
        static_conditioning=static,
        spatial_mask=mask,
    )
    shifted_static = static.copy()
    shifted_static[artifact.STATIC_ORDER.index("x"), 1, 1] += 0.125
    different_coordinates = artifact.build_transient_spatial_identity(
        sampled_source,
        reference_states=state,
        static_conditioning=shifted_static,
        spatial_mask=mask,
    )

    assert sampled_identity["evaluation_shape"] == different_indices["evaluation_shape"] == [3, 3]
    assert sampled_identity["grid_identity_sha256"] != different_indices["grid_identity_sha256"]
    assert sampled_identity["grid_identity_sha256"] != different_coordinates["grid_identity_sha256"]
    assert sampled_identity["reference_grid_identity_sha256"] == sampled_identity["prediction_grid_identity_sha256"]
    assert np.array_equal(state, state_before)
    assert np.array_equal(static, static_before)


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
            decoded_delta=next_state - state,
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
        predicted = torch.stack(states, dim=1)
        sources = torch.cat((request.state[:, None], predicted[:, :-1]), dim=1)
        decoded = predicted - sources
        return inference.TransientRolloutResult(
            states=predicted,
            scaled_deltas=torch.zeros((1, length, 4, 2, 2), dtype=torch.float32),
            timing=inference.TransientTiming(0.02 * length, "cpu", "float32", "autonomous_rollout", length),
            decoded_deltas=decoded,
            prediction_available=torch.ones((1, length), dtype=torch.bool),
        )

    monkeypatch.setattr(
        inference,
        "predict_prepared_transient_step_diagnostic",
        fake_step,
    )
    monkeypatch.setattr(
        inference,
        "rollout_prepared_transient_autonomous_diagnostic",
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
        "spatial_representation": {"schema_version": 1},
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


def test_nonfinite_evaluation_completes_all_modes_with_unavailable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep diagnostic predictions out of strict target-domain conversion."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")

    def finite_step(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
    ) -> inference.TransientStepResult:
        state = request.state.clone()
        return inference.TransientStepResult(
            next_state=state,
            scaled_delta=torch.zeros_like(state),
            decoded_delta=torch.zeros_like(state),
            timing=inference.TransientTiming(0.01, "cpu", "float32", "one_step", 1),
        )

    def nonfinite_rollout(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
    ) -> inference.TransientRolloutResult:
        shape = (1, request.length, 4, 2, 2)
        states = torch.full(shape, float("nan"), dtype=torch.float32)
        scaled = torch.full(shape, float("nan"), dtype=torch.float32)
        decoded = torch.full(shape, float("nan"), dtype=torch.float32)
        states[:, 0, 0, 0, 0] = float("nan")
        states[:, 0, 1, 0, 0] = float("inf")
        states[:, 0, 2, 0, 0] = float("-inf")
        scaled[:, 0].copy_(states[:, 0])
        decoded[:, 0].copy_(states[:, 0])
        available = torch.zeros((1, request.length), dtype=torch.bool)
        available[:, 0] = True
        return inference.TransientRolloutResult(
            states=states,
            scaled_deltas=scaled,
            decoded_deltas=decoded,
            prediction_available=available,
            timing=inference.TransientTiming(
                0.01,
                "cpu",
                "float32",
                "autonomous_rollout",
                1,
            ),
        )

    monkeypatch.setattr(
        inference,
        "predict_prepared_transient_step_diagnostic",
        finite_step,
    )
    monkeypatch.setattr(
        inference,
        "rollout_prepared_transient_autonomous_diagnostic",
        nonfinite_rollout,
    )

    evaluated = rollout.evaluate_transient_case(
        _context(),
        case,
        identity=_identity(),
        reference_completion={
            "physical_duration_hours": 3.0,
            "time_to_target_hours": None,
            "target_reached": False,
            "right_censored": True,
            "final_wet_fraction": 0.75,
            "target_wet_fraction_limit": 0.5,
            "physical_duration_availability": "available",
            "target_wet_fraction_limit_availability": "available",
        },
        target_wet_basis=0.05,
        target_fraction_limit=0.5,
        fixed_horizons=(1, 2),
    )

    diagnostic = [record for record in evaluated.records if record.prediction_validity["status"] == transient_validity.NONFINITE]
    assert {record.mode for record in diagnostic} == {
        "autonomous_full",
        "rolling_origin",
    }
    assert all(record.target["predicted_available"] is False for record in diagnostic)
    assert all(record.target["predicted_unavailable_reason"] == "prediction_contains_nonfinite_model_output" for record in diagnostic)
    full = next(record for record in diagnostic if record.mode == "autonomous_full")
    assert full.prediction_available.tolist() == [True, False, False]
    assert np.isnan(full.scaled_model_outputs[0, 0, 0, 0])
    assert np.isposinf(full.scaled_model_outputs[0, 1, 0, 0])
    assert np.isneginf(full.scaled_model_outputs[0, 2, 0, 0])
    assert evaluated.prediction_validity["status"] == transient_validity.NONFINITE
    assert evaluated.model_calls == 7


def _cache_test_record(case_id: str, *, mode: artifact.EvaluationMode) -> artifact.TransientSequenceRecord:
    """Build one minimal strict record for case-payload cache coverage."""
    static = np.zeros((len(artifact.STATIC_ORDER), 2, 2), dtype=np.float32)
    identity = _identity()
    identity["case_id"] = case_id
    identity["spatial_representation"] = _spatial_identity(static)
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
    physical_times = np.asarray([0.0, 1.0], dtype=np.float64)
    reference_states = np.zeros(
        (2, len(artifact.STATE_ORDER), 2, 2),
        dtype=np.float32,
    )
    predicted_states = reference_states.copy()
    prediction_fields = _prediction_fields(
        predicted_states=predicted_states,
        predicted_increments=np.diff(predicted_states, axis=0),
        physical_times=physical_times,
        mode=mode,
        origin_index=0,
    )
    return artifact.TransientSequenceRecord(
        mode=mode,
        case_id=case_id,
        dataset_role="id",
        origin_index=0,
        requested_horizon="full" if full_autonomous else 1,
        available_horizon=1,
        trajectory_length=2,
        physical_times=physical_times,
        transition_indices=np.asarray([0], dtype=np.int64),
        reference_states=reference_states,
        predicted_states=predicted_states,
        reference_increments=None,
        predicted_increments=prediction_fields["predicted_increments"],
        scaled_model_outputs=prediction_fields["scaled_model_outputs"],
        prediction_available=prediction_fields["prediction_available"],
        prediction_nonfinite_mask=prediction_fields["prediction_nonfinite_mask"],
        prediction_physical_invalid_mask=(prediction_fields["prediction_physical_invalid_mask"]),
        prediction_validity=prediction_fields["prediction_validity"],
        spatial_mask=np.ones((2, 2), dtype=bool),
        temporal_mask=np.ones(2, dtype=bool),
        static_conditioning=static,
        boundary_conditioning=np.zeros((1, len(artifact.BOUNDARY_ORDER)), dtype=np.float32),
        scalar_conditioning=np.zeros(len(artifact.SCALAR_ORDER), dtype=np.float32),
        identity=identity,
        target=target,
        timing={},
        exclusion={"excluded": False, "reason": None},
    ).validated()


def _nonfinite_cache_test_record(case_id: str) -> artifact.TransientSequenceRecord:
    """Build one record retaining distinct raw IEEE invalid values."""
    base = _cache_test_record(case_id, mode="autonomous_full")
    scaled_outputs = np.zeros_like(base.predicted_increments)
    scaled_outputs[0, 0, 0, 0] = np.nan
    scaled_outputs[0, 1, 0, 0] = np.inf
    scaled_outputs[0, 2, 0, 0] = -np.inf
    decoded = scaled_outputs.copy()
    predicted = base.predicted_states.copy()
    with np.errstate(invalid="ignore"):
        predicted[1] = predicted[0] + decoded[0]
    availability = np.ones(1, dtype=bool)
    target = {
        **base.target,
        "predicted_available": False,
        "predicted_unavailable_reason": ("prediction_contains_nonfinite_model_output"),
        "predicted_reached": False,
        "predicted_censored": False,
        "predicted_time_to_target": None,
        "predicted_final_gap": None,
        "predicted_final_time": None,
    }
    return replace(
        base,
        predicted_states=predicted,
        predicted_increments=decoded,
        scaled_model_outputs=scaled_outputs,
        prediction_available=availability,
        prediction_nonfinite_mask=~np.isfinite(predicted[1:]),
        prediction_physical_invalid_mask=(transient_validity.prediction_physical_invalid_mask(predicted[1:])),
        prediction_validity=transient_validity.build_prediction_validity(
            scaled_model_outputs=scaled_outputs,
            decoded_physical_increments=decoded,
            reconstructed_states=predicted[1:],
            prediction_available=availability,
            physical_times=base.physical_times,
            mode=base.mode,
            origin_index=base.origin_index,
        ),
        target=target,
    ).validated()


@pytest.mark.parametrize(
    ("channel", "value"),
    [
        ("T", -1.0),
        ("T", 2_001.0),
        ("phi", -0.01),
        ("phi", 1.01),
        ("w_surf", -0.01),
        ("w_int", -0.01),
    ],
)
def test_prediction_validity_classifies_each_finite_field_domain(
    channel: str,
    value: float,
) -> None:
    """Count finite field-domain violations without changing raw values."""
    states = np.ones((1, len(artifact.STATE_ORDER), 2, 2), dtype=np.float32)
    states[:, 0] = 300.0
    states[:, 1] = 0.5
    channel_index = artifact.STATE_ORDER.index(channel)
    states[0, channel_index, 0, 0] = value
    increments = np.zeros_like(states)

    validity = transient_validity.build_prediction_validity(
        scaled_model_outputs=increments,
        decoded_physical_increments=increments,
        reconstructed_states=states,
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )

    assert validity["status"] == transient_validity.FINITE_BUT_PHYSICALLY_INVALID
    assert validity["channels"][channel]["physically_invalid_finite_count"] == 1
    assert validity["first_invalid"]["channel"] == channel
    assert states[0, channel_index, 0, 0] == value


def test_prediction_validity_classifies_valid_and_complete_nonfinite_fields() -> None:
    """Distinguish all-valid support from one wholly non-finite state field."""
    states = np.ones((1, len(artifact.STATE_ORDER), 2, 2), dtype=np.float32)
    states[:, 0] = 300.0
    states[:, 1] = 0.5
    increments = np.zeros_like(states)
    valid = transient_validity.build_prediction_validity(
        scaled_model_outputs=increments,
        decoded_physical_increments=increments,
        reconstructed_states=states,
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )
    assert valid["status"] == transient_validity.VALID

    channel_index = artifact.STATE_ORDER.index("w_int")
    for values in (increments, states):
        values[:, channel_index] = np.nan
    nonfinite = transient_validity.build_prediction_validity(
        scaled_model_outputs=increments,
        decoded_physical_increments=increments,
        reconstructed_states=states,
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )
    assert nonfinite["status"] == transient_validity.NONFINITE
    assert nonfinite["channels"]["w_int"]["nan_count"] == 4
    assert nonfinite["stages"]["raw_scaled_model_output"]["channels"]["w_int"]["nan_count"] == 4
    assert np.isnan(states[:, channel_index]).all()


def test_prediction_validity_counts_ieee_values_and_uncomputed_tail_exactly() -> None:
    """Count only computed model outputs and retain the first exact invalid value."""
    shape = (3, len(artifact.STATE_ORDER), 1, 2)
    scaled = np.zeros(shape, dtype=np.float32)
    decoded = np.zeros(shape, dtype=np.float32)
    states = np.zeros(shape, dtype=np.float32)
    states[:, 0] = 300.0
    states[:, 1] = 0.5
    states[:, 2:] = 1.0
    for values in (scaled, decoded, states):
        values[1, 0, 0, 0] = np.nan
        values[1, 1, 0, 0] = np.inf
        values[1, 2, 0, 0] = -np.inf
        values[2] = np.nan
    validity = transient_validity.build_prediction_validity(
        scaled_model_outputs=scaled,
        decoded_physical_increments=decoded,
        reconstructed_states=states,
        prediction_available=np.asarray([True, True, False]),
        physical_times=np.asarray([0.0, 1.0, 2.0, 3.0]),
        mode="autonomous_full",
        origin_index=0,
    )

    assert validity["status"] == transient_validity.NONFINITE
    assert validity["computed_step_count"] == 2
    assert validity["uncomputed_step_count"] == 1
    assert validity["uncomputed_value_count"] == 8
    raw_channels = validity["stages"]["raw_scaled_model_output"]["channels"]
    assert raw_channels["T"]["nan_count"] == 1
    assert raw_channels["phi"]["positive_infinity_count"] == 1
    assert raw_channels["w_surf"]["negative_infinity_count"] == 1
    assert validity["first_invalid"] == {
        "kind": "NONFINITE",
        "stage": "raw_scaled_model_output",
        "mode": "autonomous_full",
        "origin_index": 0,
        "rollout_step": 2,
        "transition_index": 1,
        "physical_time": 2.0,
        "channel": "T",
        "channel_index": 0,
        "spatial_index": [0, 0],
    }
    wrong_tail = json.loads(json.dumps(validity))
    wrong_tail["uncomputed_value_count"] += 1
    with pytest.raises(ValueError, match="uncomputed count"):
        transient_validity.validate_prediction_validity_document(wrong_tail)
    wrong_first = json.loads(json.dumps(validity))
    wrong_first["first_invalid"]["transition_index"] += 1
    with pytest.raises(ValueError, match="sequence coordinates"):
        transient_validity.validate_prediction_validity_document(wrong_first)


def test_nonfinite_artifact_roundtrip_preserves_raw_values_and_unavailable_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-trip IEEE diagnostics without filtering them into ordinary metrics."""
    record = _nonfinite_cache_test_record("invalid-case")
    monkeypatch.setattr(
        artifact_generator,
        "_normalized_metric_state",
        lambda *_args, **_kwargs: pytest.fail("non-finite prediction reached ordinary metric normalization"),
    )
    statistics = artifact_generator._record_metric_statistics(
        record,
        scaling=cast("Any", object()),
    )
    for scope in ("cumulative", "endpoint"):
        assert statistics[scope]["available"] is False
        assert statistics[scope]["statistics"] is None
        assert statistics[scope]["nonfinite_value_count"] == 3

    root = tmp_path / "invalid-artifact"
    stager = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
    )
    stager.write_case(
        (record,),
        metric_statistics={record.record_id: statistics},
    )
    stager.finalize(provenance=_cache_test_provenance())

    indexed = artifact.load_transient_sequence_artifact_index(root)
    summary = indexed.summaries[0]
    assert summary.prediction_validity["status"] == transient_validity.NONFINITE
    assert summary.metric_statistics is not None
    assert summary.metric_statistics["cumulative"]["available"] is False
    loaded = artifact.load_transient_sequence_artifact(root).records[0]
    assert np.isnan(loaded.scaled_model_outputs[0, 0, 0, 0])
    assert np.isposinf(loaded.scaled_model_outputs[0, 1, 0, 0])
    assert np.isneginf(loaded.scaled_model_outputs[0, 2, 0, 0])


def test_finite_physical_invalidity_remains_eligible_for_raw_error_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate physical warnings from arithmetic availability."""
    predicted = np.zeros((2, 4, 2, 2), dtype=np.float32)
    predicted[:, 0] = 300.0
    predicted[:, 1] = 0.5
    predicted[:, 2:] = 1.0
    predicted[1, 0, 0, 0] = 2_001.0
    decoded = np.diff(predicted, axis=0)
    validity = transient_validity.build_prediction_validity(
        scaled_model_outputs=decoded,
        decoded_physical_increments=decoded,
        reconstructed_states=predicted[1:],
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )
    assert validity["status"] == (transient_validity.FINITE_BUT_PHYSICALLY_INVALID)

    def normalize(value: np.ndarray, *, scaling: object) -> np.ndarray:
        del scaling
        return np.asarray(value, dtype=np.float32)

    monkeypatch.setattr(
        artifact_generator,
        "_normalized_metric_state",
        normalize,
    )
    record = SimpleNamespace(
        predicted_states=predicted,
        reference_states=np.broadcast_to(predicted[0], predicted.shape).copy(),
        spatial_mask=np.ones((2, 2), dtype=bool),
        scalar_conditioning=np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        static_conditioning=np.ones((7, 2, 2), dtype=np.float32),
        available_horizon=1,
        prediction_available=np.ones(1, dtype=bool),
    )
    statistics = artifact_generator._record_metric_statistics(
        cast("Any", record),
        scaling=cast("Any", object()),
    )
    assert statistics["cumulative"]["available"] is True
    assert statistics["cumulative"]["statistics"] is not None


def test_artifact_record_identity_changes_with_exact_evaluation_grid() -> None:
    """Bind record identity to exact stride/index evidence, not array shape alone."""
    record = _cache_test_record("grid-case", mode="autonomous_full")
    representation = datasets.contracts.transient.resolve_spatial_representation((3, 3), 2)
    alternative_spatial = artifact.build_transient_spatial_identity(
        representation,
        reference_states=record.reference_states,
        static_conditioning=record.static_conditioning,
        spatial_mask=record.spatial_mask,
    )
    alternative_identity = {**record.identity, "spatial_representation": alternative_spatial}
    alternative = replace(record, identity=alternative_identity).validated()

    assert record.identity["spatial_representation"]["evaluation_spatial_stride"] == 1
    assert alternative.identity["spatial_representation"]["evaluation_spatial_stride"] == 2
    assert record.reference_states.shape == alternative.reference_states.shape
    assert record.record_id != alternative.record_id


def test_artifact_rejects_reference_prediction_grid_identity_disagreement() -> None:
    """Fail closed when prediction grid evidence diverges from its reference."""
    record = _cache_test_record("grid-case", mode="autonomous_full")
    spatial = dict(record.identity["spatial_representation"])
    spatial["prediction_grid_identity_sha256"] = "f" * 64
    identity = {**record.identity, "spatial_representation": spatial}

    with pytest.raises(artifact.TransientSequenceArtifactError, match="Reference and prediction"):
        replace(record, identity=identity).validated()


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
        "spatial_representation": {"schema_version": 1},
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


def test_resumable_stager_restores_completed_case_from_strict_manifest(tmp_path: Path) -> None:
    """Reuse a completed case after an interrupted stage without rewriting its bundle."""
    root = tmp_path / "resumable"
    identity = {"run": "run-a", "role": "id", "config": "a" * 64}
    first = (
        _cache_test_record("resume-a", mode="autonomous_full"),
        _cache_test_record("resume-a", mode="teacher_forced_one_step"),
    )
    second = (
        _cache_test_record("resume-b", mode="autonomous_full"),
        _cache_test_record("resume-b", mode="teacher_forced_one_step"),
    )
    initial = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    initial.write_case(first, resume_evidence={"timing": {"forward_calls": 4}})
    first_payload = root / artifact._case_payload_path("resume-a")
    first_digest = artifact.common.serialization.file_sha256(first_payload)
    interrupted_payload = root / "npz" / ".interrupted.partial.npz"
    interrupted_payload.write_bytes(b"incomplete")

    resumed = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    assert resumed.completed_case_ids == {"resume-a"}
    assert resumed.completed_case_evidence("resume-a") == {"timing": {"forward_calls": 4}}
    assert artifact.common.serialization.file_sha256(first_payload) == first_digest
    assert not interrupted_payload.exists()
    resumed.write_case(second)
    resumed.finalize(provenance=_cache_test_provenance())

    assert artifact_contracts.artifact_provenance_path(root).is_file()
    assert not (root / artifact.TransientSequenceArtifactStager._RESUME_DESCRIPTOR).exists()
    assert not (root / artifact.TransientSequenceArtifactStager._CASE_MANIFEST_DIRECTORY).exists()


def test_resumable_stager_rejects_self_digested_semantic_payload_tamper(
    tmp_path: Path,
) -> None:
    """Rebind resumed manifest claims to the raw arrays before reuse."""
    root = tmp_path / "tampered-resume"
    identity = {"run": "run-a", "role": "id", "config": "a" * 64}
    records = (
        _cache_test_record("resume-a", mode="autonomous_full"),
        _cache_test_record("resume-a", mode="teacher_forced_one_step"),
    )
    initial = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    initial.write_case(records, resume_evidence={"case": "resume-a"})
    payload_path = root / artifact._case_payload_path("resume-a")
    with np.load(payload_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    scaled_name = next(name for name in payload if name.endswith("_scaled_model_outputs"))
    payload[scaled_name] = payload[scaled_name].copy()
    payload[scaled_name][0, 0, 0, 0] = np.nan
    np.savez(payload_path, **payload)
    digest = common.serialization.file_sha256(payload_path)
    manifest_path = next((root / artifact.TransientSequenceArtifactStager._CASE_MANIFEST_DIRECTORY).glob("case-*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload"]["sha256"] = digest
    for row in manifest["rows"]:
        row["payload_sha256"] = digest
    common.serialization.atomic_write_json(manifest_path, manifest)

    with pytest.raises(
        artifact.TransientSequenceArtifactError,
        match=r"payload semantics conflict.*--rebuild",
    ):
        artifact.TransientSequenceArtifactStager(
            root,
            dataset_name="transient_dataset",
            dataset_role="id",
            resume_identity=identity,
        )


def test_resumed_case_evidence_restores_json_timing_collections() -> None:
    """Restore tuple-valued timing evidence after its durable JSON round trip."""
    case_id = "resume-json-case"
    record = _cache_test_record(case_id, mode="autonomous_full")
    prediction_validity = transient_validity.aggregate_case_prediction_validity(
        case_id=case_id,
        records=(record.prediction_validity,),
    )
    timing_case = artifact_generator.transient_timing.TransientTimingCase(
        case_id=case_id,
        repetitions={"drying_no_rollout_model_seconds": (0.25,)},
        device="cpu",
        precision="float32",
        dataset_backend="canonical_hdf5",
        warmup_passes=1,
    )
    identity = {
        "case_id": case_id,
        "material_family": "lentil",
        "simulation_identity": {"package_case_id": case_id},
        "dataset_role": "id",
        "spatial_representation": {"grid": "exact"},
    }
    evidence = json.loads(
        json.dumps(
            {
                "schema_version": artifact_generator._RESUME_EVIDENCE_SCHEMA_VERSION,
                "case_id": case_id,
                "material_family": "lentil",
                "identity": identity,
                "spatial_identity": identity["spatial_representation"],
                "component_availability": {
                    "prediction_validity": prediction_validity,
                    "timing_case": artifact_generator.asdict(timing_case),
                },
                "timing_case": artifact_generator.asdict(timing_case),
                "prediction_validity_records": [record.prediction_validity],
                "spatial_compatibility": {
                    "architecture": {"decision": "SUPPORTED_EXACTLY"},
                    "scaling": {"decision": "SUPPORTED_EXACTLY"},
                },
                "progress": {"rollout_steps": 1},
            }
        )
    )

    restored = artifact_generator._restore_resumed_case_evidence(
        evidence,
        case_id=case_id,
        expected_material="lentil",
        dataset_role="id",
    )

    assert restored.timing_case.repetitions == {"drying_no_rollout_model_seconds": (0.25,)}
    assert restored.prediction_validity == prediction_validity


def test_resumable_stager_rejects_symbolic_staging_root(tmp_path: Path) -> None:
    """Refuse resume writes through symbolic staging-root indirection."""
    target = tmp_path / "target"
    target.mkdir()
    symbolic = tmp_path / "resume-link"
    symbolic.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be symbolic"):
        artifact.TransientSequenceArtifactStager(
            symbolic,
            dataset_name="transient_dataset",
            dataset_role="id",
            resume_identity={"run": "run-a"},
        )


def test_service_requires_exact_rebuild_for_conflicting_partial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never replace incomplete canonical evidence without explicit rebuild."""
    run_dir = tmp_path / "transient_drying" / "runs" / "synthetic-b"
    save_root = run_dir / "analysis" / "id"
    save_root.mkdir(parents=True)
    (save_root / "partial.marker").write_text("partial\n", encoding="utf-8")
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=(("qualified-case",),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="synthetic-b",
        id_role=role,
        ood_role=None,
    )
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "validate_completed_run",
        lambda _run_dir: {"config": {}, "normalizer_state": {}},
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "transient_artifact_plan_from_completed",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "generate_transient_role_artifact",
        lambda **_kwargs: pytest.fail("conflicting target reached generation"),
    )

    with pytest.raises(artifact_service.ArtifactCacheError, match="--rebuild"):
        artifact_service._run_or_load_transient_artifacts_locked(
            run_dir=run_dir,
            dataset_name="transient_dataset",
            split="eval",
            device_resolution=cast(
                "Any",
                SimpleNamespace(device=torch.device("cpu")),
            ),
            dataset_root=tmp_path / "datasets",
            rebuild=False,
            evaluation_spatial_stride=1,
        )


def test_all_completed_generator_resume_skips_runtime_owners(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalize exact staged cases without reopening any inference-time owner."""
    case_id = "resume-case"
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=((case_id,),),
        membership_digests=("a" * 64,),
    )
    representation = SimpleNamespace(
        source_shape=(2, 2),
        represented_shape=(2, 2),
        spatial_stride=1,
        as_dict=lambda: {"shape": [2, 2], "spatial_stride": 1},
    )
    completed = {
        "config": {"evaluation": {}, "training": {}},
        "scientific_run_name": "resume-run",
        "effective_config_digest": "b" * 64,
        "checkpoint_identity": {"checkpoint": "c" * 64},
        "selected_checkpoint_sha256": "c" * 64,
        "selected_checkpoint_epoch": 1,
        "normalizer_sha256": "d" * 64,
    }
    identity = {
        "case_id": case_id,
        "material_family": "lentil",
        "simulation_identity": {"package_case_id": case_id},
        "dataset_role": "id",
        "spatial_representation": {"grid": "exact"},
    }
    resumed = artifact_generator._ResumedCaseEvidence(
        identity=identity,
        spatial_identity={"grid": "exact"},
        component_availability={},
        timing_case=cast("Any", object()),
        prediction_validity={"status": "VALID"},
        spatial_compatibility={"architecture": {"decision": "SUPPORTED_EXACTLY"}, "scaling": {"decision": "SUPPORTED_EXACTLY"}},
        material="lentil",
        rollout_steps=2,
    )
    finalized: list[dict[str, Any]] = []

    class Stager:
        completed_case_ids = frozenset({case_id})

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def completed_case_evidence(self, value: str) -> dict[str, Any]:
            assert value == case_id
            return {"stored": True}

        def finalize(self, *, provenance: dict[str, Any]) -> object:
            finalized.append(provenance)
            return object()

    monkeypatch.setattr(artifact_generator, "_require_completed_transient", lambda value: value)
    monkeypatch.setattr(
        artifact_generator, "resolve_transient_artifact_spatial_representations", lambda *_args, **_kwargs: (representation, representation)
    )
    monkeypatch.setattr(artifact_generator.sequence_artifact, "TransientSequenceArtifactStager", Stager)
    monkeypatch.setattr(artifact_generator, "_restore_resumed_case_evidence", lambda *_args, **_kwargs: resumed)
    monkeypatch.setattr(artifact_generator.transient_timing, "build_transient_timing_report", lambda _cases: {"report": "restored"})
    monkeypatch.setattr(artifact_generator, "asdict", lambda value: cast("dict[str, Any]", value))
    monkeypatch.setattr(
        artifact_generator,
        "_role_prediction_validity_evidence",
        lambda *_args, **_kwargs: {"status_counts": {"VALID": 1, "FINITE_BUT_PHYSICALLY_INVALID": 0, "NONFINITE": 0}},
    )
    monkeypatch.setattr(artifact_generator, "_role_spatial_representation_evidence", lambda **_kwargs: {"grid": "restored"})
    monkeypatch.setattr(
        artifact_generator,
        "_role_provenance",
        lambda **_kwargs: {"run": {}, "dataset": {}, "evaluation": {}, "spatial_representation": {"grid": "restored"}, "runtime": {}, "lineage": {}},
    )
    for name in ("build_transient_inference_context",):
        monkeypatch.setattr(
            artifact_generator.learning.inference.transient,
            name,
            lambda *_args, **_kwargs: pytest.fail("all-completed resume constructed inference context"),
        )
    monkeypatch.setattr(artifact_generator, "_load_role_datasets", lambda *_args, **_kwargs: pytest.fail("all-completed resume loaded Dataset"))
    monkeypatch.setattr(
        artifact_generator, "_generation_case_sources", lambda *_args, **_kwargs: pytest.fail("all-completed resume loaded generation sources")
    )
    monkeypatch.setattr(artifact_generator, "_materialize_case", lambda *_args, **_kwargs: pytest.fail("all-completed resume materialized a case"))

    result = artifact_generator.generate_transient_role_artifact(
        run_dir=tmp_path / "run",
        role=role,
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"), as_dict=dict)),
        dataset_root=tmp_path / "02_datasets" / "packages",
        staging_root=tmp_path / "stage",
        completed=completed,
    )

    assert result is not None
    assert len(finalized) == 1


def test_service_retries_finalized_resume_stage_without_generator_or_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish a completed private stage through finalization-only retry."""
    run_dir = tmp_path / "transient_drying" / "runs" / "synthetic-b"
    run_dir.mkdir(parents=True)
    save_root = run_dir / "analysis" / "id"
    staging_root = save_root.parent / f".{save_root.name}.transient-resume"
    marker = artifact_contracts.artifact_provenance_path(staging_root)
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    resume_descriptor = staging_root / artifact.TransientSequenceArtifactStager._RESUME_DESCRIPTOR
    resume_descriptor.write_text("{}\n", encoding="utf-8")
    resume_manifests = staging_root / artifact.TransientSequenceArtifactStager._CASE_MANIFEST_DIRECTORY
    resume_manifests.mkdir()
    (resume_manifests / "case-interrupted.json").write_text("{}\n", encoding="utf-8")
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=(("qualified-case",),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="synthetic-b",
        id_role=role,
        ood_role=None,
    )
    completed = {
        "config": {
            "task": "transient_drying",
            "training": {"stage": "b"},
        },
        "scientific_run_name": "synthetic-b",
        "normalizer_state": {"schema_version": 1},
    }
    summary = SimpleNamespace(
        case_id="qualified-case",
        identity={"material_family": "lentil"},
    )
    frame = pd.DataFrame({"record_id": ["record"]})
    staged = SimpleNamespace(
        summaries=(summary,),
        case_ids=("qualified-case",),
        frame=frame,
        provenance={"kind": "staged"},
        unavailable_horizons=(),
    )
    reporter = artifact_performance.ArtifactProgressReporter(
        task="transient_drying",
        run="synthetic-b",
        stage_label="b",
        checkpoint_label="best",
        device=torch.device("cpu"),
        dtype="float32",
        total_cases=1,
        split="id",
        output_root=save_root,
    )
    published_snapshots: list[dict[str, Any]] = []
    admitted_roots: list[Path] = []
    monkeypatch.setattr(
        artifact_service.experiments.run,
        "validate_completed_run",
        lambda _run_dir: completed,
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "transient_artifact_plan_from_completed",
        lambda *_args, **_kwargs: plan,
    )
    monkeypatch.setattr(
        artifact_service,
        "_transient_progress_reporter",
        lambda **_kwargs: reporter,
    )

    def admit(root: Path, **_kwargs: Any) -> Any:
        admitted_roots.append(root)
        return staged

    monkeypatch.setattr(
        artifact_service.transient,
        "validate_transient_role_artifact_index",
        admit,
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "generate_transient_role_artifact",
        lambda **_kwargs: pytest.fail("finalization-only retry invoked artifact generation or inference"),
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "validate_staged_transient_role_artifact",
        lambda index, **_kwargs: index,
    )

    def publish(**_kwargs: Any) -> None:
        final_marker = artifact_contracts.artifact_provenance_path(save_root)
        final_marker.parent.mkdir(parents=True, exist_ok=True)
        final_marker.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        artifact_service,
        "_publish_staged_artifact",
        publish,
    )
    monkeypatch.setattr(
        artifact,
        "publish_transient_operational_performance",
        lambda _root, snapshot: published_snapshots.append(dict(snapshot)),
    )
    monkeypatch.setattr(artifact_service, "cleanup_runtime", lambda _device: None)

    result = artifact_service._run_or_load_transient_artifacts_locked(
        run_dir=run_dir,
        dataset_name="transient_dataset",
        split="eval",
        device_resolution=cast("Any", SimpleNamespace(device=torch.device("cpu"))),
        dataset_root=tmp_path / "datasets",
        rebuild=False,
        evaluation_spatial_stride=1,
    )

    assert result is frame
    assert admitted_roots == [staging_root, save_root]
    assert not resume_descriptor.exists()
    assert not resume_manifests.exists()
    assert len(published_snapshots) == 1
    counts = published_snapshots[0]["counts"]
    assert counts["case_count"] == 1
    assert counts["reused_case_count"] == 1
    assert counts["forward_call_count"] == 0
    assert counts["timed_forward_call_count"] == 0


def test_resumable_stager_rejects_incompatible_partial_identity(tmp_path: Path) -> None:
    """Refuse a partial stage whose exact scientific identity no longer matches."""
    root = tmp_path / "incompatible-resume"
    artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity={"run": "old"},
    )
    with pytest.raises(artifact.TransientSequenceArtifactError, match="--rebuild"):
        artifact.TransientSequenceArtifactStager(
            root,
            dataset_name="transient_dataset",
            dataset_role="id",
            resume_identity={"run": "new"},
        )


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


@pytest.mark.parametrize(
    ("stage_label", "comparison_arm", "stage_key"),
    [
        ("a0", None, "stage_a0"),
        ("a", "a0", "stage_a0"),
        ("b", "b", "stage_b"),
        ("a", "a_plus", "stage_a_plus"),
    ],
)
def test_parent_experiment_uses_full_resolved_config_hash_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage_label: str,
    comparison_arm: str | None,
    stage_key: str,
) -> None:
    """Bind historical/current A, B, and A+ children in the full hash domain."""
    task = "transient_drying"
    run_name = f"synthetic-{comparison_arm or stage_label}"
    run_dir = tmp_path / task / "runs" / run_name
    run_dir.mkdir(parents=True)
    marker = tmp_path / "parent" / "experiment.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    config = {
        "task": task,
        "run": {"name": run_name},
        "training": {
            "stage": stage_label,
            "epochs": 7,
            **({} if comparison_arm is None else {"comparison_arm": comparison_arm}),
        },
        "evaluation": {"metrics": [{"id": "metric"}]},
    }
    resolved_digest = artifact_generator.run_identity.resolved_config_digest(config)
    checkpoint_scope_digest = "e" * 64
    assert resolved_digest != checkpoint_scope_digest
    record = {
        "parent_label": "synthetic-parent",
        "parent_identity_sha256": "a" * 64,
        "run_revision": 0,
        "source_repository": {"commit": None, "dirty": None},
        "children": {
            stage_key: {
                "path": str(run_dir),
                "run_name": run_name,
                "resolved_config_sha256": resolved_digest,
            }
        },
    }
    monkeypatch.setattr(
        artifact_generator.run_identity,
        "experiment_record_path",
        lambda *_args, **_kwargs: marker,
    )
    monkeypatch.setattr(
        artifact_generator.run_identity,
        "validate_transient_experiment_record",
        lambda _value: record,
    )
    completed = {
        "run_dir": run_dir,
        "summary": {
            "run_identity": {
                "parent_label": "synthetic-parent",
                "source_repository": {"commit": None, "dirty": None},
            }
        },
        "config": config,
        "scientific_run_name": run_name,
        "effective_config_digest": checkpoint_scope_digest,
    }

    evidence = artifact_generator._parent_experiment_evidence(completed)

    assert evidence["kind"] == "grouped"
    assert evidence["parent_identity_sha256"] == "a" * 64
    record["children"][stage_key]["resolved_config_sha256"] = checkpoint_scope_digest
    with pytest.raises(ValueError, match="child identity contradicts"):
        artifact_generator._parent_experiment_evidence(completed)


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
        training_spatial_shape=(2, 2),
        evaluation_spatial_shape=(2, 2),
        architecture_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
        scaling_spatial_compatibility={"decision": "SUPPORTED_EXACTLY"},
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
        expected_model_calls=3,
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

    truncated = artifact_generator._timing_case(
        case_id="case",
        benchmark=benchmark,
        expected_model_calls=4,
        generation_timing={},
        dataset_backend="pt_shards",
        pt_identity=None,
        runtime_metadata={
            "resolved_device": "cpu",
            "processor": "test-cpu",
            "cuda_device_name": None,
            "python_version": "test",
            "pytorch_version": "test",
        },
    )
    assert "drying_no_rollout_model_seconds" not in truncated.repetitions
    assert "drying_no_end_to_end_seconds" not in truncated.repetitions
    assert truncated.unavailable_reasons["drying_no_rollout_model_seconds"] == "diagnostic_rollout_stopped_after_nonfinite_prediction"


def test_artifact_rejects_unsupported_model_alias() -> None:
    """Reject hidden UNO-RNO aliases at the sequence admission boundary."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")
    identity = _identity(model_kind="uno_rno")
    predicted_states = case.reference_states[:2]
    prediction_fields = _prediction_fields(
        predicted_states=predicted_states,
        predicted_increments=np.diff(predicted_states, axis=0),
        physical_times=case.physical_times[:2],
        mode="teacher_forced_one_step",
        origin_index=0,
    )
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
            predicted_states=predicted_states,
            reference_increments=np.diff(case.reference_states[:2], axis=0),
            predicted_increments=prediction_fields["predicted_increments"],
            scaled_model_outputs=prediction_fields["scaled_model_outputs"],
            prediction_available=prediction_fields["prediction_available"],
            prediction_nonfinite_mask=(prediction_fields["prediction_nonfinite_mask"]),
            prediction_physical_invalid_mask=(prediction_fields["prediction_physical_invalid_mask"]),
            prediction_validity=prediction_fields["prediction_validity"],
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
