"""Protect transient Evaluation sequence, rollout, and persistence contracts.

Responsibilities:
  - Check exact membership, rollout modes, validity, resumable staging, and sequence persistence

Design principles:
  - Identity binds source and evaluation grids; raw invalid predictions retain truthful evidence

This module does NOT:
  - Train models or authorize canonical-source cleanup
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from support import transient_evaluation
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

# Strict package case IDs name the per-case NPZ files.
_CASE_ID = "lentil__case_0001__000000000001"
_CASE_B = "lentil__case_0002__000000000002"


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
    result: list[dict[str, Any]] = [
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
                "sample_id": f"{_CASE_ID}__step_{index:04d}",
                "sample_mode": "one_step_transition",
                "rollout_length": 1,
                "simulation_case_id": _CASE_ID,
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
        for index in range(3)
    ]
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
        "case_id": _CASE_ID,
        "dataset_identity": {"dataset_id": "transient_dataset", "index_digest": "a" * 64},
        "dataset_role": "id",
        "evaluation_assignment": None,
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


@pytest.mark.parametrize("missing_metadata", [None, "meta", "simulation_profile", "template_sha256"])
def test_base_identity_distinguishes_qualified_package_and_local_generation_case(
    monkeypatch: pytest.MonkeyPatch,
    missing_metadata: str | None,
) -> None:
    """Use admitted scientific metadata alongside the immutable Source selector."""
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
        },
        "meta": {
            "simulation_profile": "transient_drying",
            "template_sha256": "4" * 64,
        },
    }
    if missing_metadata is not None:
        if missing_metadata == "meta":
            canonical.pop("meta")
        else:
            canonical["meta"].pop(missing_metadata)
        # Source-selector extensions cannot substitute for absent admitted metadata.
        canonical["source"].update(simulation_profile="transient_drying", template_sha256="4" * 64)
        with pytest.raises(KeyError, match=missing_metadata):
            artifact_generator._base_identity(
                completed=completed,
                role=role,
                case=cast("Any", case),
                canonical=canonical,
                dataset_backend="canonical_hdf5",
                pt_identity=None,
                runtime_identity={"scientific_solver_seconds": 1.0},
                spatial_representation=_spatial_identity(),
            )
        return

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
    assert identity["simulation_identity"]["simulation_profile"] == "transient_drying"
    assert identity["simulation_identity"]["template_sha256"] == "4" * 64
    assert "simulation_profile" not in canonical["source"]
    assert "template_sha256" not in canonical["source"]


def test_transient_artifact_reopens_exact_manifest_source_without_terminal_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Bind a selected package case to its exact Source selector and design."""
    package_case_id = "transient_drying__lentil__natural__case_0001"
    source = {
        "schema_kind": "generation_source_case_reference",
        "schema_version": 1,
        "task": "transient_drying",
        "campaign_name": "campaign-1",
        "campaign_digest": "a" * 64,
        "material_family": "lentil",
        "case_id": "case_0001",
        "case_input_id": "b" * 64,
        "simulation_case_id": "c" * 64,
        "case_hdf5_sha256": "d" * 64,
        "source_batch_id": "batch-1",
        "case_hdf5_relative": "01_generation/transient_drying/campaign-1/lentil/case_0001/case.h5",
    }
    design = {"origin": "replacement", "candidate_position": 2}
    manifest = {"source_case_identities": [{"package_case_id": package_case_id, "source": source, "design": design}]}
    admitted = SimpleNamespace(source_identity=source, design_provenance=design)
    calls: list[dict[str, Any]] = []

    def load_evidence(task: str, dataset_id: str, **_kwargs: Any) -> dict[str, Any]:
        if (task, dataset_id) != ("transient_drying", "transient_dataset"):
            pytest.fail("wrong package")
        return manifest

    monkeypatch.setattr(artifact_generator.datasets.packages, "load_package_manifest_evidence", load_evidence)

    def resolve(_task: str, selector: dict[str, Any], **_kwargs: Any) -> Any:
        calls.append(selector)
        return admitted

    monkeypatch.setattr(artifact_generator.generation.publication.catalog, "resolve_raw_case_reference", resolve)
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
    assert sources == {package_case_id: admitted}
    assert calls == [{key: source[key] for key in artifact_generator._SOURCE_SELECTOR_KEYS}]

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
    monkeypatch.setattr(
        artifact_generator.generated_batch,
        "interpret_admitted_transient_source_case",
        lambda value, **_kwargs: canonical if value is admitted else pytest.fail("wrong admitted case"),
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
    case_record = {
        "source_hdf5_sha256": source["case_hdf5_sha256"],
        "case_input_id": source["case_input_id"],
        "simulation_case_id": source["simulation_case_id"],
        "source_batch_id": source["source_batch_id"],
        "source_relative_path": source["case_hdf5_relative"],
    }
    interpreted, runtime = artifact_generator._canonical_case_evidence(
        evaluation_case,
        package_case_id=package_case_id,
        case_record=case_record,
        generation_sources=sources,
    )
    assert interpreted is canonical
    assert runtime == canonical["runtime"]
    with pytest.raises(ValueError, match="contradicts its admitted Source identity"):
        artifact_generator._canonical_case_evidence(
            evaluation_case,
            package_case_id=package_case_id,
            case_record={**case_record, "simulation_case_id": "different"},
            generation_sources=sources,
        )


def test_transient_artifact_rejects_manifest_source_drift(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reject a current Dataset whose saved source design no longer re-admits."""
    package_case_id = _CASE_ID
    source = dict.fromkeys(artifact_generator._SOURCE_SELECTOR_KEYS, "a" * 64)
    manifest = {"source_case_identities": [{"package_case_id": package_case_id, "source": source, "design": {"origin": "primary"}}]}
    monkeypatch.setattr(artifact_generator.datasets.packages, "load_package_manifest_evidence", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        artifact_generator.generation.publication.catalog,
        "resolve_raw_case_reference",
        lambda *_args, **_kwargs: SimpleNamespace(source_identity=source, design_provenance={"origin": "replacement"}),
    )
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="dataset",
        source_dataset_ids=("dataset",),
        source_identities=({},),
        case_ids_by_source=((package_case_id,),),
        membership_digests=("a" * 64,),
    )
    with pytest.raises(ValueError, match="design evidence drifted"):
        artifact_generator._generation_case_sources(role, storage_root=tmp_path)


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


@pytest.mark.parametrize(("with_additional", "primary_ood"), [(False, False), (True, False), (True, True)])
def test_current_transient_plan_uses_saved_exact_membership_and_optional_additional(
    tmp_path: Path, *, with_additional: bool, primary_ood: bool
) -> None:
    """Preserve one current split schema and an ordinary empty additional selection."""
    evaluation_id = "train-dataset" if primary_ood else "additional-dataset"
    additional_ids = [evaluation_id] if with_additional and not primary_ood else []
    assignments = (
        [{"dataset_id": evaluation_id, "role": "ood", "relation": "Dataset-declared material holdout", "assumptions": []}] if with_additional else []
    )
    completed = {
        "config": {
            "task": "transient_drying",
            "run": {"name": "scientific-run"},
            "data": {"train_dataset": "train-dataset", "additional_evaluation_datasets": additional_ids},
        },
        "split_indices": {
            "schema_kind": datasets.runtime.transient_training.SPLIT_SCHEMA_KIND,
            "schema_version": datasets.runtime.transient_training.SPLIT_SCHEMA_VERSION,
            "dataset_identity": {
                "train": {"dataset_id": "train-dataset", "index_digest": "a" * 64},
                "additional_evaluation": [{"dataset_id": evaluation_id, "index_digest": "b" * 64}] if with_additional else [],
            },
            "roles": {
                "id_test": {"case_ids": ["train-case"], "item_ids": ["train-case__step_0000"], "membership_digest": "c" * 64},
                "additional_evaluation": {
                    "parts": [{"case_ids": ["additional-case"], "item_ids": ["additional-case__step_0000"], "membership_digest": "d" * 64}]
                    if with_additional
                    else []
                },
            },
            "evaluation_assignments": assignments,
        },
    }
    plan = artifact_generator.transient_artifact_plan_from_completed(completed, run_dir=tmp_path)
    assert plan.id_role.case_ids_by_source == (("train-case",),)
    assert (plan.additional_role is not None) is with_additional
    if with_additional:
        assert plan.additional_role is not None
        assert plan.additional_role.split == "additional_evaluation"
        assert plan.additional_role.evaluation_assignments == tuple(assignments)


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
            additional_evaluation_dataset_name=None,
            scientific_run_name="run",
        ),
        device_resolution=cast("Any", object()),
        id_frame=pd.DataFrame(),
    )


@pytest.mark.parametrize("with_additional", [False, True])
def test_transient_artifact_publication_logs_only_validated_bounded_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    with_additional: bool,
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
    if with_additional:
        frame_by_role["additional_evaluation"] = pd.DataFrame()
        provenance_by_role["additional_evaluation"] = {"role": "additional_evaluation", "identity": "b" * 64}
    for role, frame in frame_by_role.items():
        frame.attrs["artifact_provenance"] = provenance_by_role[role]

    def artifact_root(*, split: str, **_kwargs: Any) -> Any:
        return tmp_path / split

    def validate(*, artifact_root: Any, **_kwargs: Any) -> dict[str, Any]:
        role = "id" if artifact_root.name == "eval" else "additional_evaluation"
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
        task="transient_drying",
        additional_evaluation_dataset_name="additional_dataset" if with_additional else None,
        scientific_run_name="run",
    )
    artifact_service._upload_published_artifacts(
        plan=plan,
        device_resolution=cast("Any", object()),
        id_frame=frame_by_role["id"],
        additional_frame=frame_by_role.get("additional_evaluation"),
    )

    expected_roles = ["id", "additional_evaluation"] if with_additional else ["id"]
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
            additional_evaluation_dataset_name="ood_dataset",
            scientific_run_name="run",
        ),
        device_resolution=cast("Any", object()),
        id_frame=frame,
        additional_frame=frame,
    )

    assert len(file_uploads) == 2
    assert len(post_uploads) == 1
    assert post_uploads[0]["artifact_root"] == tmp_path / "eval"
    assert events[-1] == "finish_wandb"


def test_multi_package_role_rejects_duplicate_package_case_identity() -> None:
    """Prevent additional Evaluation package parts from colliding in sequence and timing identities."""
    identities = (
        {"dataset_id": "additional_a"},
        {"dataset_id": "additional_b"},
    )
    evidence = (
        {"case_ids": ["case"], "membership_digest": "a" * 64},
        {"case_ids": ["case"], "membership_digest": "b" * 64},
    )
    with pytest.raises(ValueError, match="duplicate package-case"):
        artifact_generator._role_plan(
            split="additional_evaluation",
            dataset_role="additional_evaluation",
            dataset_ids=("additional_a", "additional_b"),
            identities=identities,
            role_evidence=evidence,
        )


def test_scoped_role_selection_preserves_saved_order_and_rejects_unknown_cases() -> None:
    """Keep disposable case selection bounded to exact saved role membership."""
    role = artifact_generator.TransientArtifactRolePlan(
        split="additional_evaluation",
        dataset_role="additional_evaluation",
        dataset_name="combined-additional",
        source_dataset_ids=("additional-a", "additional-b"),
        source_identities=(
            {"dataset_id": "additional-a"},
            {"dataset_id": "additional-b"},
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
    [("a", "a0"), ("b", "b")],
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
        split="additional_evaluation",
        evaluation_spatial_stride=2,
    )

    assert canonical == run_dir / "analysis" / "id"
    assert sampled == run_dir / "analysis" / "grid_s2" / "id"
    assert sampled_ood == run_dir / "analysis" / "grid_s2" / "additional_evaluation" / "dataset"
    assert artifact_service._artifact_lock_path(run_dir=run_dir, save_root=canonical) != artifact_service._artifact_lock_path(
        run_dir=run_dir,
        save_root=sampled,
    )


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
        case_ids_by_source=((_CASE_ID,),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="run",
        id_role=role,
        additional_role=None,
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


def test_rollout_modes_origins_horizons_conditioning_and_sequence_roundtrip(  # noqa: PLR0915 -- one compact rollout-to-artifact test verifies the complete persistence boundary
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

    def fake_teacher_forced(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
        *,
        reference_states: torch.Tensor,
        capture: Any = (),
    ) -> inference.TransientTeacherForcedResult:
        # The fixture context is an RNO, so every rolling origin's warm state is requested.
        assert tuple(capture) == (0, 1, 2)
        next_states = reference_states.clone()
        next_states[:, :, 0] += 1.0
        next_states[:, :, 2:] -= 1.0
        return inference.TransientTeacherForcedResult(
            next_states=next_states,
            scaled_deltas=torch.zeros_like(reference_states),
            decoded_deltas=next_states - reference_states,
            step_seconds=(0.01,) * request.length,
            timing=inference.TransientTiming(0.01 * request.length, "cpu", "float32", "one_step", request.length),
            captured_hidden=dict.fromkeys(capture),
        )

    def fake_rollout(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
        *,
        initial_hidden: object | None = None,
    ) -> inference.TransientRolloutResult:
        assert initial_hidden is None
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
        "teacher_forced_prepared_transient_diagnostic",
        fake_teacher_forced,
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
    assert all(value > 0.0 for value in benchmark.warmed_prepared_rollout_wall_seconds)
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
                "kind": "grouped",
                "parent_available": True,
                "parent_label": "transient_parent",
                "parent_identity_sha256": "a" * 64,
                "source_repository": {"commit": None, "dirty": None},
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

    def counted_loader(row: Any, payload: Any) -> artifact.TransientSequenceRecord:
        calls.append(str(row["record_id"]))
        return original_loader(row, payload)

    monkeypatch.setattr(artifact, "_record_from_row", counted_loader)
    indexed = artifact.load_transient_sequence_artifact_index(tmp_path / "artifact")
    payload_path = tmp_path / "artifact" / persisted_rows["payload_path"].iloc[0]
    assert indexed.case_ids == (_CASE_ID,)
    assert indexed.cache_size == 0
    assert indexed.cache_limit == 1
    assert calls == []
    artifact.validate_transient_sequence_payload_inventory(indexed)
    assert payload_loads == [str(payload_path)]
    assert indexed.cache_size == 0
    payload_loads.clear()
    selected_case = indexed.records(case_id=_CASE_ID)
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
            expected_cases={_CASE_ID},
            dataset_role="id",
        )


def test_nonfinite_evaluation_completes_all_modes_with_unavailable_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep diagnostic predictions out of strict target-domain conversion."""
    case = rollout.assemble_transient_evaluation_case(_items(), dataset_role="id")

    def finite_teacher_forced(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
        *,
        reference_states: torch.Tensor,
        capture: Any = (),
    ) -> inference.TransientTeacherForcedResult:
        del capture
        return inference.TransientTeacherForcedResult(
            next_states=reference_states.clone(),
            scaled_deltas=torch.zeros_like(reference_states),
            decoded_deltas=torch.zeros_like(reference_states),
            step_seconds=(0.01,) * request.length,
            timing=inference.TransientTiming(0.01 * request.length, "cpu", "float32", "one_step", request.length),
            captured_hidden={},
        )

    def nonfinite_rollout(
        _context: inference.TransientInferenceContext,
        request: inference.TransientPreparedRequest,
        *,
        initial_hidden: object | None = None,
    ) -> inference.TransientRolloutResult:
        del initial_hidden
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
        "teacher_forced_prepared_transient_diagnostic",
        finite_teacher_forced,
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
    assert full.scaled_model_outputs is not None
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
        ("phi", 1.11),
        ("w_surf", -0.01),
        ("w_int", -0.01),
    ],
)
def test_prediction_validity_classifies_each_finite_field_domain(
    channel: str,
    value: float,
) -> None:
    """Count finite field-domain violations without changing raw values."""
    states = np.ones((1, len(artifact.STATE_ORDER), 6, 6), dtype=np.float32)
    states[:, 0] = 300.0
    states[:, 1] = 0.5
    channel_index = artifact.STATE_ORDER.index(channel)
    states[0, channel_index, 2, 2] = value
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
    assert states[0, channel_index, 2, 2] == value


def _validity_states(grid: int = 6) -> np.ndarray:
    states = np.ones((1, len(artifact.STATE_ORDER), grid, grid), dtype=np.float32)
    states[:, 0] = 300.0
    states[:, 1] = 0.5
    return states


def _status_of(states: np.ndarray) -> str:
    zeros = np.zeros_like(states)
    return transient_validity.build_prediction_validity(
        scaled_model_outputs=zeros,
        decoded_physical_increments=zeros,
        reconstructed_states=states,
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )["status"]


@pytest.mark.parametrize(
    ("phi", "index", "status"),
    [
        (1.09, (2, 2), transient_validity.VALID),
        (5.6, (0, 0), transient_validity.VALID),
        (-0.64, (1, 5), transient_validity.VALID),
        (5.6, (2, 0), transient_validity.FINITE_BUT_PHYSICALLY_INVALID),
    ],
)
def test_physical_invalid_mask_phi_range_and_corner_blocks(phi: float, index: tuple[int, int], status: str) -> None:
    """Apply the phi range [0, 1.10] and exclude the 2x2 corner blocks."""
    states = _validity_states()
    states[0, 1, index[0], index[1]] = phi
    assert _status_of(states) == status


def test_corner_nan_remains_nonfinite() -> None:
    """Corner exclusion does not hide non-finite values."""
    states = _validity_states()
    states[0, 1, 0, 0] = np.nan
    assert _status_of(states) == transient_validity.NONFINITE


def test_prediction_validity_rejects_schema_version_one() -> None:
    """Readers reject the superseded validity schema."""
    states = _validity_states()
    zeros = np.zeros_like(states)
    validity = transient_validity.build_prediction_validity(
        scaled_model_outputs=zeros,
        decoded_physical_increments=zeros,
        reconstructed_states=states,
        prediction_available=np.ones(1, dtype=bool),
        physical_times=np.asarray([0.0, 1.0]),
        mode="autonomous_full",
        origin_index=0,
    )
    assert validity["physical_contract"]["excluded_corner_block"] == 2
    with pytest.raises(ValueError, match="schema_version 1 is unsupported"):
        transient_validity.validate_prediction_validity_document({**validity, "schema_version": 1})


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
    record = _nonfinite_cache_test_record(_CASE_ID)
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
    assert loaded.scaled_model_outputs is None
    assert np.array_equal(loaded.prediction_nonfinite_mask, record.prediction_nonfinite_mask)
    assert np.array_equal(loaded.prediction_physical_invalid_mask, record.prediction_physical_invalid_mask)
    assert np.array_equal(loaded.predicted_increments, np.diff(record.predicted_states, axis=0), equal_nan=True)
    assert loaded.reference_increments is not None
    assert np.array_equal(loaded.reference_increments, np.diff(record.reference_states, axis=0))
    with np.load(root / indexed.summaries[0].payload_path, allow_pickle=False) as archive:
        members = set(archive.files)
    chain = indexed.summaries[0].chain_id
    assert {name for name in members if name.startswith("chain_")} == {
        f"chain_{chain}_predicted_states",
        f"chain_{chain}_prediction_available",
    }
    assert "reference_increments" not in members


def _loaded_record(tmp_path: Path, record: artifact.TransientSequenceRecord) -> artifact.TransientSequenceRecord:
    """Persist one record and return it as loaded without raw scaled outputs."""
    root = tmp_path / "loaded"
    stager = artifact.TransientSequenceArtifactStager(root, dataset_name="transient_dataset", dataset_role="id")
    stager.write_case((record,))
    stager.finalize(provenance=_cache_test_provenance())
    loaded = artifact.load_transient_sequence_artifact(root).records[0]
    assert loaded.scaled_model_outputs is None
    return loaded


def test_write_admission_requires_raw_scaled_outputs(tmp_path: Path) -> None:
    """Fresh writes validate against raw outputs before the reduced bundle drops them."""
    record = replace(_cache_test_record(_CASE_ID, mode="autonomous_full"), scaled_model_outputs=None)
    stager = artifact.TransientSequenceArtifactStager(tmp_path / "raw", dataset_name="transient_dataset", dataset_role="id")
    with pytest.raises(artifact.TransientSequenceArtifactError, match="raw scaled"):
        stager.write_case((record,))


def test_load_admission_rejects_impossible_first_invalid(tmp_path: Path) -> None:
    """First-invalid evidence must stay inside the persisted time and grid support."""
    loaded = _loaded_record(tmp_path, _nonfinite_cache_test_record(_CASE_ID))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    validity["first_invalid"]["spatial_index"] = [999, 999]
    validity["first_invalid"]["physical_time"] = 999.0
    with pytest.raises(ValueError, match="first-invalid"):
        replace(loaded, prediction_validity=validity).validated()


def test_load_admission_rejects_fabricated_nonfinite_on_finite_payload(tmp_path: Path) -> None:
    """Raw-stage non-finite counts require non-finite decoded and state values."""
    loaded = _loaded_record(tmp_path, _cache_test_record(_CASE_ID, mode="autonomous_full"))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    channel = validity["stages"]["raw_scaled_model_output"]["channels"][artifact.STATE_ORDER[0]]
    channel.update(finite_value_count=channel["finite_value_count"] - 1, nonfinite_value_count=1, nan_count=1)
    validity["status"] = transient_validity.NONFINITE
    validity["first_invalid"] = {
        "kind": transient_validity.NONFINITE,
        "stage": "raw_scaled_model_output",
        "mode": validity["mode"],
        "origin_index": validity["origin_index"],
        "rollout_step": 1,
        "transition_index": validity["origin_index"],
        "physical_time": float(loaded.physical_times[1]),
        "channel": artifact.STATE_ORDER[0],
        "channel_index": 0,
        "spatial_index": [0, 0],
    }
    with pytest.raises(ValueError, match="propagation"):
        replace(loaded, prediction_validity=validity).validated()


def test_load_admission_rejects_upstream_first_invalid_on_a_finite_state(tmp_path: Path) -> None:
    """A raw or decoded non-finite value must reach the persisted state at the same element."""
    loaded = _loaded_record(tmp_path, _nonfinite_cache_test_record(_CASE_ID))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    assert validity["first_invalid"]["stage"] != "reconstructed_state"
    channel = validity["first_invalid"]["channel_index"]
    y_index, x_index = (int(item) for item in np.argwhere(np.isfinite(loaded.predicted_states[1, channel]))[0])
    validity["first_invalid"]["spatial_index"] = [y_index, x_index]
    with pytest.raises(ValueError, match="first-invalid evidence"):
        replace(loaded, prediction_validity=validity).validated()


@pytest.mark.parametrize(
    ("stage", "channel_index"),
    [("reconstructed_state", 0), ("raw_scaled_model_output", 1)],
)
def test_load_admission_rejects_first_invalid_after_the_earliest_persisted_nan(tmp_path: Path, stage: str, channel_index: int) -> None:
    """A persisted NaN was a raw NaN, so first-invalid evidence cannot sort after it."""
    loaded = _loaded_record(tmp_path, _nonfinite_cache_test_record(_CASE_ID))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    validity["first_invalid"].update(stage=stage, channel=artifact.STATE_ORDER[channel_index], channel_index=channel_index)
    with pytest.raises(ValueError, match="first-invalid evidence"):
        replace(loaded, prediction_validity=validity).validated()


def test_load_admission_rejects_nan_claimed_where_the_state_is_infinite(tmp_path: Path) -> None:
    """NaN counts agree across stages; overflow may only add infinities of one sign."""
    loaded = _loaded_record(tmp_path, _nonfinite_cache_test_record(_CASE_ID))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    for stage in ("raw_scaled_model_output", "decoded_physical_increment"):
        channel = validity["stages"][stage]["channels"][artifact.STATE_ORDER[1]]
        assert channel["positive_infinity_count"] == 1
        channel.update(nan_count=1, positive_infinity_count=0)
    with pytest.raises(ValueError, match="propagation"):
        replace(loaded, prediction_validity=validity).validated()


def test_load_admission_rejects_step_counts_contradicting_availability(tmp_path: Path) -> None:
    """Requested and computed step counts must match the persisted availability mask."""
    loaded = _loaded_record(tmp_path, _nonfinite_cache_test_record(_CASE_ID))
    validity = json.loads(json.dumps(loaded.prediction_validity))
    validity["requested_step_count"] = 2
    validity["uncomputed_step_count"] = 1
    validity["uncomputed_value_count"] = len(artifact.STATE_ORDER) * int(np.prod(loaded.predicted_states.shape[-2:]))
    with pytest.raises(ValueError, match="step counts"):
        replace(loaded, prediction_validity=validity).validated()


@pytest.mark.parametrize("old_version", [1, 2, 3, 4])
def test_superseded_schema_versions_are_rejected(tmp_path: Path, old_version: int) -> None:
    """Reject artifacts written before bundles dropped redundant arrays or were named by package case ID."""
    root = tmp_path / "old-artifact"
    stager = artifact.TransientSequenceArtifactStager(root, dataset_name="transient_dataset", dataset_role="id")
    stager.write_case((_cache_test_record(_CASE_ID, mode="autonomous_full"),))
    stager.finalize(provenance=_cache_test_provenance())
    marker_path = root / artifact_contracts.ARTIFACT_PROVENANCE_FILENAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_schema_version"] = old_version
    common.serialization.atomic_write_json(marker_path, marker)
    with pytest.raises(artifact.TransientSequenceArtifactError, match=f"schema version {old_version} is unsupported"):
        artifact.load_transient_sequence_artifact_index(root)


def _chain_test_record(
    mode: artifact.EvaluationMode,
    *,
    origin_index: int,
    horizon: int,
    reference: np.ndarray,
    chain: np.ndarray,
    requested: artifact.RequestedHorizon | None = None,
) -> artifact.TransientSequenceRecord:
    """Build one strict record that follows ``reference[origin_index]`` with the first ``horizon`` states of ``chain``."""
    base = _cache_test_record(_CASE_ID, mode=mode)
    states = np.concatenate((reference[origin_index : origin_index + 1], chain[:horizon]))
    times = np.arange(origin_index, origin_index + horizon + 1, dtype=np.float64)
    fields = _prediction_fields(
        predicted_states=states,
        predicted_increments=np.diff(states, axis=0),
        physical_times=times,
        mode=mode,
        origin_index=origin_index,
    )
    target = {**base.target, "predicted_final_time": float(times[-1])}
    if mode == "autonomous_full":
        target["reference_final_time"] = float(times[-1])
    return replace(
        base,
        origin_index=origin_index,
        requested_horizon=horizon if requested is None else requested,
        available_horizon=horizon,
        trajectory_length=len(reference),
        physical_times=times,
        transition_indices=np.arange(origin_index, origin_index + horizon, dtype=np.int64),
        reference_states=reference[origin_index : origin_index + horizon + 1],
        predicted_states=states,
        predicted_increments=fields["predicted_increments"],
        scaled_model_outputs=fields["scaled_model_outputs"],
        prediction_available=fields["prediction_available"],
        prediction_nonfinite_mask=fields["prediction_nonfinite_mask"],
        prediction_physical_invalid_mask=fields["prediction_physical_invalid_mask"],
        prediction_validity=fields["prediction_validity"],
        temporal_mask=np.ones(horizon + 1, dtype=bool),
        boundary_conditioning=np.zeros((horizon, len(artifact.BOUNDARY_ORDER)), dtype=np.float32),
        target=target,
    ).validated()


def _chain_test_case() -> tuple[tuple[np.ndarray, ...], tuple[artifact.TransientSequenceRecord, ...]]:
    """Return the distinct chains of a four-state case and its records; the full autonomous and origin-0 rolling chains are bit-identical."""
    grid = (len(artifact.STATE_ORDER), 2, 2)
    reference = np.arange(4 * 16, dtype=np.float32).reshape(4, *grid) / np.float32(200.0)
    full_chain = reference[1:] + np.float32(0.01)
    rolling_chain = reference[2:] + np.float32(0.02)
    one_step_chain = reference[3:] + np.float32(0.03)
    first_step_chain = reference[1:2] + np.float32(0.04)
    records = (
        _chain_test_record("autonomous_full", origin_index=0, horizon=3, reference=reference, chain=full_chain, requested="full"),
        _chain_test_record("rolling_origin", origin_index=0, horizon=3, reference=reference, chain=full_chain, requested="full"),
        _chain_test_record("rolling_origin", origin_index=0, horizon=1, reference=reference, chain=full_chain),
        _chain_test_record("rolling_origin", origin_index=1, horizon=2, reference=reference, chain=rolling_chain),
        _chain_test_record("teacher_forced_one_step", origin_index=0, horizon=1, reference=reference, chain=first_step_chain),
        _chain_test_record("teacher_forced_one_step", origin_index=2, horizon=1, reference=reference, chain=one_step_chain),
    )
    return (full_chain, rolling_chain, one_step_chain, first_step_chain), records


def test_chain_storage_omits_first_states_and_duplicate_chains_and_roundtrips_bit_for_bit(tmp_path: Path) -> None:
    """Store each distinct chain once without its reference first state and rebuild every record exactly."""
    chains, records = _chain_test_case()
    root = _write_chain_test_artifact(tmp_path, records)

    rows = pd.read_parquet(root / "transient_dataset.parquet")
    chain_ids = rows.groupby(["mode", "origin_index"])["chain_id"].agg(lambda values: tuple(set(values)))
    assert all(len(values) == 1 for values in chain_ids)
    assert chain_ids["autonomous_full", 0] == chain_ids["rolling_origin", 0]
    assert len({values[0] for values in chain_ids}) == len(chains)
    with np.load(next((root / "npz").glob("*.npz")), allow_pickle=False) as archive:
        stored = [archive[name] for name in archive.files if name.endswith("_predicted_states")]
    assert len(stored) == len(chains)
    assert all(any(np.array_equal(states, chain) for states in stored) for chain in chains)

    restored = {record.record_id: record for record in artifact.load_transient_sequence_artifact(root).records}
    assert set(restored) == {record.record_id for record in records}
    for record in records:
        loaded = restored[record.record_id]
        assert loaded.predicted_states.dtype == np.float32
        assert np.array_equal(loaded.predicted_states.view(np.uint32), record.predicted_states.view(np.uint32))
        assert np.array_equal(loaded.reference_states.view(np.uint32), record.reference_states.view(np.uint32))
        assert np.array_equal(loaded.prediction_available, record.prediction_available)


def test_chain_storage_writer_fails_closed_when_the_rebuild_would_differ(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Refuse to publish a bundle whose stored chains would not rebuild every record bit for bit."""
    _chains, records = _chain_test_case()
    monkeypatch.setattr(artifact, "_chain_content_key", lambda _record: b"every chain looks identical")
    stager = artifact.TransientSequenceArtifactStager(tmp_path / "wrong", dataset_name="transient_dataset", dataset_role="id")
    with pytest.raises(artifact.TransientSequenceArtifactError, match="bit for bit"):
        stager.write_case(records)


def _write_chain_test_artifact(tmp_path: Path, records: tuple[artifact.TransientSequenceRecord, ...]) -> Path:
    """Write the chain fixture as a finalized artifact and return its root."""
    root = tmp_path / "chains"
    artifact.write_transient_sequence_artifact(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=records,
        provenance=_cache_test_provenance(),
    )
    return root


def _resign_chain_test_artifact(root: Path, *, rows: pd.DataFrame | None = None, payload: dict[str, np.ndarray] | None = None) -> None:
    """Replace the summary table and/or the case bundle, then rebind the row digests and the output manifest."""
    payload_path = next((root / "npz").glob("*.npz"))
    if payload is not None:
        cast("Any", np.savez)(payload_path, **payload)
    parquet_path = root / "transient_dataset.parquet"
    table = pd.read_parquet(parquet_path) if rows is None else rows
    table["payload_sha256"] = common.serialization.file_sha256(payload_path)
    table.to_parquet(parquet_path, index=False)
    marker_path = artifact_contracts.artifact_provenance_path(root)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["outputs"] = artifact_contracts.artifact_output_manifest(root)
    common.serialization.atomic_write_json(marker_path, marker)


@pytest.mark.parametrize(
    ("victims", "victim_origin", "owner"),
    [
        pytest.param(("teacher_forced_one_step",), 2, None, id="unknown-owner"),
        pytest.param(("rolling_origin",), 1, ("autonomous_full", 0), id="owner-at-another-origin"),
        pytest.param(("autonomous_full", "rolling_origin"), 0, ("teacher_forced_one_step", 0), id="owner-shorter-than-alias"),
    ],
)
def test_load_rejects_chain_storage_without_a_valid_owner(
    tmp_path: Path, victims: tuple[str, ...], victim_origin: int, owner: tuple[str, int] | None
) -> None:
    """Admit a stored chain only from an owner of its case with the same origin that covers its longest horizon."""
    _chains, records = _chain_test_case()
    root = _write_chain_test_artifact(tmp_path, records)
    rows = pd.read_parquet(root / "transient_dataset.parquet")
    owner_chain = "0" * 24 if owner is None else rows.loc[(rows["mode"] == owner[0]) & (rows["origin_index"] == owner[1]), "chain_id"].iloc[0]
    rows.loc[rows["mode"].isin(victims) & (rows["origin_index"] == victim_origin), "chain_id"] = owner_chain
    _resign_chain_test_artifact(root, rows=rows)
    with pytest.raises(artifact.TransientSequenceArtifactError, match="prediction-chain"):
        artifact.load_transient_sequence_artifact_index(root)


def test_load_rejects_a_stored_chain_longer_than_its_longest_record(tmp_path: Path) -> None:
    """Reject an owner chain carrying an appended state and availability flag."""
    _chains, records = _chain_test_case()
    root = _write_chain_test_artifact(tmp_path, records)
    with np.load(next((root / "npz").glob("*.npz")), allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    states = max((name for name in payload if name.endswith("_predicted_states")), key=lambda name: len(payload[name]))
    available = states.replace("_predicted_states", "_prediction_available")
    payload[states] = np.concatenate((payload[states], payload[states][-1:]))
    payload[available] = np.append(payload[available], True)
    _resign_chain_test_artifact(root, payload=payload)
    index = artifact.load_transient_sequence_artifact_index(root)
    with pytest.raises(artifact.TransientSequenceArtifactError, match="stored chain length"):
        index.records()
    index.close()


def test_records_of_one_stored_chain_share_one_read_only_rebuilt_array(tmp_path: Path) -> None:
    """Rebuild each stored chain once per open case and hand its records read-only views of it."""
    _chains, records = _chain_test_case()
    root = _write_chain_test_artifact(tmp_path, records)
    index = artifact.load_transient_sequence_artifact_index(root)
    loaded = {(record.mode, record.origin_index, record.requested_horizon): record for record in index.records()}
    full = loaded["autonomous_full", 0, "full"].predicted_states
    assert np.shares_memory(full, loaded["rolling_origin", 0, "full"].predicted_states)
    assert np.shares_memory(full, loaded["rolling_origin", 0, 1].predicted_states)
    assert not np.shares_memory(full, loaded["rolling_origin", 1, 2].predicted_states)
    for array in (full, loaded["rolling_origin", 0, 1].reference_states, loaded["rolling_origin", 0, 1].prediction_available):
        with pytest.raises(ValueError, match="read-only"):
            array.fill(array.flat[0])
    index.close()


def test_finite_physical_invalidity_remains_eligible_for_raw_error_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate physical warnings from arithmetic availability."""
    predicted = np.zeros((2, 4, 2, 2), dtype=np.float32)
    predicted[:, 0] = 300.0
    predicted[:, 1] = 0.5
    predicted[:, 2:] = 1.0
    predicted[1, 0, 0, 0] = 2_001.0
    interior = np.pad(predicted, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="edge")
    decoded = np.diff(interior, axis=0)
    validity = transient_validity.build_prediction_validity(
        scaled_model_outputs=decoded,
        decoded_physical_increments=decoded,
        reconstructed_states=interior[1:],
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
    record = _cache_test_record(_CASE_ID, mode="autonomous_full")
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
    record = _cache_test_record(_CASE_ID, mode="autonomous_full")
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
                "kind": "grouped",
                "parent_available": True,
                "parent_label": "transient_parent",
                "parent_identity_sha256": "a" * 64,
                "source_repository": {"commit": None, "dirty": None},
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
        _cache_test_record(_CASE_ID, mode="autonomous_full"),
        _cache_test_record(_CASE_ID, mode="teacher_forced_one_step"),
    )
    second = (
        _cache_test_record(_CASE_B, mode="autonomous_full"),
        _cache_test_record(_CASE_B, mode="teacher_forced_one_step"),
    )

    stager.write_case(first)

    marker = artifact_contracts.artifact_provenance_path(root)
    first_payload = root / artifact._case_payload_path(_CASE_ID)
    assert first_payload.is_file()
    assert not marker.exists()
    assert len(stager._rows) == 2
    assert not any(isinstance(value, np.ndarray) for row in stager._rows for value in row.values())

    stager.write_case(second)
    index = stager.finalize(provenance=_cache_test_provenance())

    assert marker.is_file()
    assert index.case_ids == (_CASE_ID, _CASE_B)
    assert sorted(path.name for path in (root / "npz").glob("*.npz")) == [f"{_CASE_ID}.npz", f"{_CASE_B}.npz"]
    assert observed_hashes.count(first_payload) == 1
    assert len([path for path in observed_hashes if path.suffix == ".npz"]) == 2
    assert len([path for path in observed_hashes if path.suffix == ".parquet"]) == 1
    for payload in (root / "npz").glob("*.npz"):
        with zipfile.ZipFile(payload) as archive:
            assert {entry.compress_type for entry in archive.infolist()} == {zipfile.ZIP_STORED}


def test_stager_rejects_a_case_id_that_is_not_a_package_case_id(tmp_path: Path) -> None:
    """Refuse to name a case bundle after anything but a strict package case ID."""
    root = tmp_path / "artifact"
    stager = artifact.TransientSequenceArtifactStager(root, dataset_name="transient_dataset", dataset_role="id")

    with pytest.raises(artifact.TransientSequenceArtifactError, match="package case ID"):
        stager.write_case((_cache_test_record("not-a-package-case", mode="autonomous_full"),))
    assert not any((root / "npz").iterdir())


def test_load_rejects_a_case_bundle_named_by_another_case(tmp_path: Path) -> None:
    """Admit a case bundle only under the package case ID its summary rows declare."""
    _chains, records = _chain_test_case()
    root = _write_chain_test_artifact(tmp_path, records)
    original = root / "npz" / f"{_CASE_ID}.npz"
    renamed = original.rename(original.with_name(f"{_CASE_B}.npz"))
    rows = pd.read_parquet(root / "transient_dataset.parquet")
    rows["payload_path"] = f"npz/{renamed.name}"
    _resign_chain_test_artifact(root, rows=rows)

    with pytest.raises(artifact.TransientSequenceArtifactError, match="contradicts its exact case identity"):
        artifact.load_transient_sequence_artifact_index(root)


def test_resumable_stager_restores_completed_case_from_strict_manifest(tmp_path: Path) -> None:
    """Reuse a completed case after an interrupted stage without rewriting its bundle."""
    root = tmp_path / "resumable"
    identity = {"run": "run-a", "role": "id", "config": "a" * 64}
    first = (
        _cache_test_record(_CASE_ID, mode="autonomous_full"),
        _cache_test_record(_CASE_ID, mode="teacher_forced_one_step"),
    )
    second = (
        _cache_test_record(_CASE_B, mode="autonomous_full"),
        _cache_test_record(_CASE_B, mode="teacher_forced_one_step"),
    )
    initial = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    initial.write_case(first, resume_evidence={"timing": {"forward_calls": 4}})
    first_payload = root / artifact._case_payload_path(_CASE_ID)
    first_digest = artifact.common.serialization.file_sha256(first_payload)
    interrupted_payload = root / "npz" / ".interrupted.partial.npz"
    interrupted_payload.write_bytes(b"incomplete")

    resumed = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    assert resumed.completed_case_ids == {_CASE_ID}
    assert resumed.completed_case_evidence(_CASE_ID) == {"timing": {"forward_calls": 4}}
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
        _cache_test_record(_CASE_ID, mode="autonomous_full"),
        _cache_test_record(_CASE_ID, mode="teacher_forced_one_step"),
    )
    initial = artifact.TransientSequenceArtifactStager(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        resume_identity=identity,
    )
    initial.write_case(records, resume_evidence={"case": _CASE_ID})
    payload_path = root / artifact._case_payload_path(_CASE_ID)
    with np.load(payload_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    states_name = next(name for name in payload if name.endswith("_predicted_states"))
    payload[states_name] = payload[states_name].copy()
    payload[states_name][0, 0, 0, 0] = np.nan
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
    case_id = _CASE_ID
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
                "schema_kind": artifact_generator._RESUME_EVIDENCE_SCHEMA_KIND,
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
        case_ids_by_source=((_CASE_ID,),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="synthetic-b",
        id_role=role,
        additional_role=None,
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
    case_id = _CASE_ID
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
        dataset_root=tmp_path / "02_datasets",
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
        case_ids_by_source=((_CASE_ID,),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="synthetic-b",
        id_role=role,
        additional_role=None,
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
        case_id=_CASE_ID,
        identity={"material_family": "lentil"},
    )
    frame = pd.DataFrame({"record_id": ["record"]})
    staged = SimpleNamespace(
        summaries=(summary,),
        case_ids=(_CASE_ID,),
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
        case_ids_by_source=((_CASE_ID,),),
        membership_digests=("a" * 64,),
    )
    malformed = _cache_test_provenance()
    malformed["evaluation"]["scope"] = {"kind": "full"}
    malformed_root = tmp_path / "malformed-scoped"
    artifact.write_transient_sequence_artifact(
        malformed_root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(_cache_test_record(_CASE_ID, mode="autonomous_full"),),
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
        "selected_case_ids": [_CASE_ID],
        "saved_case_ids_by_source": [[_CASE_ID]],
    }
    scoped_root = tmp_path / "valid-scoped"
    artifact.write_transient_sequence_artifact(
        scoped_root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(_cache_test_record(_CASE_ID, mode="autonomous_full"),),
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

    assert admitted.case_ids == (_CASE_ID,)


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
            _cache_test_record(_CASE_ID, mode="autonomous_full"),
            _cache_test_record(
                _CASE_ID,
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

    assert admitted.record(admitted.record_ids[0]).case_id == _CASE_ID
    with zipfile.ZipFile(payload_path) as archive:
        assert zipfile.ZIP_DEFLATED in {entry.compress_type for entry in archive.infolist()}


def test_transient_artifact_index_lazily_opens_case_payloads_and_closes_lru_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """Open one selected case archive lazily, reuse it, and close it on eviction."""
    first_case = _CASE_ID
    second_case = _CASE_B
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
    assert digest_paths == []  # the digest recorded when this unchanged file was written is reused
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


def test_artifact_files_are_hashed_once_across_admission_and_case_opens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hash the Parquet table and each case bundle once, however often the artifact is admitted and opened."""
    root = tmp_path / "artifact"
    artifact.write_transient_sequence_artifact(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(_cache_test_record(_CASE_ID, mode="autonomous_full"), _cache_test_record(_CASE_B, mode="autonomous_full")),
        provenance=_cache_test_provenance(),
    )
    monkeypatch.setattr(artifact_contracts, "_FILE_DIGESTS", {})
    original = common.serialization.file_sha256
    hashed: list[Path] = []

    def counted(value: Path | str) -> str:
        hashed.append(Path(value))
        return original(value)

    monkeypatch.setattr(common.serialization, "file_sha256", counted)
    for _ in range(2):
        index = artifact.load_transient_sequence_artifact_index(root)
        index.records()
        index.close()

    assert sorted(hashed) == sorted([root / "transient_dataset.parquet", *(root / "npz").glob("*.npz")])


def test_cached_file_sha256_rehashes_when_path_size_mtime_or_inode_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reuse a digest only while the file identity is unchanged."""
    monkeypatch.setattr(artifact_contracts, "_FILE_DIGESTS", {})
    hashed: list[Path] = []
    original = common.serialization.file_sha256

    def counted(value: Path | str) -> str:
        hashed.append(Path(value))
        return original(value)

    monkeypatch.setattr(common.serialization, "file_sha256", counted)

    def rewrite(path: Path, content: bytes, mtime_ns: int) -> None:
        path.write_bytes(content)
        os.utime(path, ns=(mtime_ns, mtime_ns))

    payload = tmp_path / "payload.bin"
    rewrite(payload, b"aaaa", 1_000_000_000)
    assert artifact_contracts.cached_file_sha256(payload) == artifact_contracts.cached_file_sha256(payload) == hashlib.sha256(b"aaaa").hexdigest()
    assert len(hashed) == 1
    rewrite(payload, b"aaaaa", 1_000_000_000)  # size
    assert artifact_contracts.cached_file_sha256(payload) == hashlib.sha256(b"aaaaa").hexdigest()
    rewrite(payload, b"bbbbb", 2_000_000_000)  # modification time
    assert artifact_contracts.cached_file_sha256(payload) == hashlib.sha256(b"bbbbb").hexdigest()
    replacement = tmp_path / "replacement.bin"
    rewrite(replacement, b"ccccc", 2_000_000_000)
    replacement.replace(payload)  # inode
    assert artifact_contracts.cached_file_sha256(payload) == hashlib.sha256(b"ccccc").hexdigest()
    assert len(hashed) == 4


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
            "source_dataset_id": f"dataset_{material}",
            "index_digest": material.ljust(64, "a")[:64],
        }
        identity["evaluation_assignment"] = (
            None
            if role == "id"
            else {
                "dataset_id": f"dataset_{material}",
                "role": "ood",
                "relation": "Dataset-declared material holdout",
                "assumptions": [],
            }
        )
        return replace(
            record,
            dataset_role=role,
            identity=identity,
        ).validated()

    id_root = tmp_path / "id"
    ood_root = tmp_path / "additional_evaluation"
    id_provenance = _cache_test_provenance()
    id_provenance["dataset"] = {
        "name": "dataset_lentil_chickpea",
        "role": "id",
        "identity": {"index_digest": "a" * 64},
    }
    ood_provenance = _cache_test_provenance()
    ood_provenance["dataset"] = {
        "name": "dataset_kidney_bean",
        "role": "additional_evaluation",
        "identity": {"index_digest": "f" * 64},
    }
    artifact.write_transient_sequence_artifact(
        id_root,
        dataset_name="dataset_lentil_chickpea",
        dataset_role="id",
        records=(
            role_record(
                "chickpea__case_0051__000000000051",
                role="id",
                material="chickpea",
            ),
        ),
        provenance=id_provenance,
    )
    artifact.write_transient_sequence_artifact(
        ood_root,
        dataset_name="dataset_kidney_bean",
        dataset_role="additional_evaluation",
        records=(
            role_record(
                "kidney_bean__case_0004__000000000004",
                role="additional_evaluation",
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


@pytest.mark.parametrize("stage", ["a0", "single"])
@pytest.mark.parametrize("relocated", [False, True])
def test_independent_transient_artifact_binds_exact_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    *,
    relocated: bool,
) -> None:
    """Admit independent current bundles by exact scientific identity without parent placement."""
    config = {
        "task": "transient_drying",
        "run": {
            "name": "scientific-run",
            "layout": {
                "task": "transient_drying",
                "dataset": "dataset",
                "family": "experiment",
                "config_name": "fno_m2x2_h4_l1",
                "model_label": "fno_m2x2_h4_l1",
                "seed": 9,
                "revision": None,
                "budget_epochs": None,
                "stage": stage,
                "pair": None,
                "trial": None,
                "a0_id": "a" * 16,
                "branch_id": None,
            },
        },
        "paths": {"output_root": str(tmp_path)},
    }
    monkeypatch.setattr(artifact_generator.learning.training.checkpoint, "resume_contract_digest", common.serialization.canonical_json_sha256)
    completed = {
        "run_dir": tmp_path / "relocated-independent" if relocated else common.paths.resolve_config_run_dir(config),
        "summary": {},
        "config": config,
        "scientific_run_name": "scientific-run",
        "checkpoint_identity": {"resume_contract_digest": common.serialization.canonical_json_sha256(config)},
    }
    evidence = artifact_generator._parent_experiment_evidence(completed)
    assert evidence["kind"] == "independent"
    assert evidence["semantic_config_sha256"] == common.serialization.canonical_json_sha256(config)
    assert artifact._validate_parent_experiment_evidence(evidence) == evidence
    changed = {**evidence, "semantic_config_sha256": "invalid"}
    with pytest.raises(artifact.TransientSequenceArtifactError, match="identity is invalid"):
        artifact._validate_parent_experiment_evidence(changed)


def test_parent_experiment_provenance_requires_complete_branch_evidence() -> None:
    """Require exact complete branch evidence when a child belongs to B/A1."""
    grouped = artifact._validate_parent_experiment_evidence(
        {
            "kind": "grouped",
            "parent_available": True,
            "parent_label": "transient_parent",
            "parent_identity_sha256": "b" * 64,
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
        cold_prepared_rollout_wall_seconds=10.0,
        warmed_model_seconds=(1.0, 2.0, 3.0),
        warmed_prepared_rollout_wall_seconds=(2.0, 3.0, 4.0),
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
    assert "drying_no_prepared_rollout_wall_seconds" not in truncated.repetitions
    assert truncated.unavailable_reasons["drying_no_rollout_model_seconds"] == "diagnostic_rollout_stopped_after_nonfinite_prediction"


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
        case_ids_by_source=((_CASE_ID,),),
        membership_digests=("a" * 64,),
    )
    plan = artifact_generator.TransientArtifactPlan(
        run_dir=run_dir,
        run_name="scientific-run",
        id_role=role,
        additional_role=None,
    )
    monkeypatch.setattr(
        artifact_service.transient,
        "load_transient_artifact_plan",
        lambda _path: plan,
    )

    with pytest.raises(ValueError, match="outside the run"):
        artifact_service.build_scoped_transient_artifact(
            run_dir=run_dir,
            dataset_root=tmp_path / "02_datasets",
            output_root=run_dir / "debug-artifact",
            split="id",
            one_case=True,
        )


@pytest.mark.parametrize(
    "simulation",
    [
        {"package_case_id": _CASE_ID, "generation_case_id": "case_0001"},
        {"package_case_id": "wrong-case"},
        None,
        [],
    ],
)
def test_role_validator_binds_exact_package_identity(
    simulation: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admit qualified membership and reject missing or conflicting simulation evidence."""
    case_id = _CASE_ID
    role = artifact_generator.TransientArtifactRolePlan(
        split="eval",
        dataset_role="id",
        dataset_name="transient_dataset",
        source_dataset_ids=("transient_dataset",),
        source_identities=({"dataset_id": "transient_dataset"},),
        case_ids_by_source=((case_id,),),
        membership_digests=("a" * 64,),
    )
    completed = {
        "scientific_run_name": "test-run",
        "effective_config_digest": "b" * 64,
        "checkpoint_identity": {"task": "transient_drying"},
        "selected_checkpoint_sha256": "c" * 64,
        "selected_checkpoint_epoch": 1,
        "normalizer_sha256": "d" * 64,
        "config": {"evaluation": {}},
    }
    protocol_identity = artifact.evaluation_protocol_identity({})
    record = SimpleNamespace(
        case_id=case_id,
        identity={
            "simulation_identity": simulation,
            "evaluation_config_identity": protocol_identity,
        },
    )
    predicted_validity = {"cases": {case_id: {}}}
    provenance = {
        "dataset": {
            "source_dataset_ids": list(role.source_dataset_ids),
            "source_identities": [dict(value) for value in role.source_identities],
            "case_ids_by_source": [list(values) for values in role.case_ids_by_source],
            "membership_digests": list(role.membership_digests),
            "evaluation_assignments": [dict(value) for value in role.evaluation_assignments],
        },
        "run": {
            "name": completed["scientific_run_name"],
            "effective_config_digest": completed["effective_config_digest"],
            "checkpoint_identity": completed["checkpoint_identity"],
            "best_checkpoint_sha256": completed["selected_checkpoint_sha256"],
            "best_checkpoint_epoch": completed["selected_checkpoint_epoch"],
            "normalizer_sha256": completed["normalizer_sha256"],
            "parent_experiment": {},
        },
        "lineage": {},
        "evaluation": {
            "prediction_validity": predicted_validity,
            "config_identity": protocol_identity,
            "metrics": None,
            "objective": None,
            "modes": list(artifact.EVALUATION_MODES),
            "fixed_horizons": list(artifact.FIXED_HORIZONS),
            "rolling_origin_policy": artifact.ROLLING_ORIGIN_POLICY,
            "recurrent_state_policy": artifact.RECURRENT_STATE_POLICY,
            "process_diagnostic_policy": artifact_generator._process_diagnostic_policy(),
            "pipeline_analysis": {
                "conditions": ["A_comsol_reference", "B_drying_on_comsol_airflow", "C_drying_on_airflow_no"],
                "c_available": False,
                "fabricated_prediction_count": 0,
                "c_unavailable_reasons": ["test-owned unavailable airflow surrogate"],
            },
            "component_availability": {case_id: {"prediction_validity": {}}},
            "timing_report": {},
        },
    }
    loaded = SimpleNamespace(
        dataset_name=role.dataset_name,
        dataset_role=role.dataset_role,
        provenance=provenance,
        summaries=(record,),
        unavailable_horizons=(),
    )
    # Independently covered physical/protocol services are inert here. The complete
    # role-evidence validator and its package-membership checks execute unchanged.
    monkeypatch.setattr(artifact_generator, "_parent_experiment_evidence", lambda _completed: {})
    monkeypatch.setattr(artifact_generator, "_lineage", lambda _completed: {})
    monkeypatch.setattr(artifact_generator, "_validate_role_spatial_representation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(artifact_generator, "_validate_sequence_inventory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(artifact_generator, "_role_prediction_validity_from_records", lambda *_args, **_kwargs: predicted_validity)
    monkeypatch.setattr(
        artifact_generator.transient_timing,
        "admit_transient_timing_report",
        lambda _value: SimpleNamespace(cases=(SimpleNamespace(case_id=case_id),)),
    )
    if isinstance(simulation, dict) and simulation.get("package_case_id") == case_id:
        artifact_generator._validate_transient_role_evidence(cast("Any", loaded), completed=completed, role=role)
    else:
        with pytest.raises(ValueError, match="qualified Dataset package identity"):
            artifact_generator._validate_transient_role_evidence(cast("Any", loaded), completed=completed, role=role)


def test_role_validator_rejects_an_optuna_trial_before_binding_any_evidence() -> None:
    """A trial keeps no checkpoints, so evidence validation names that instead of failing on missing checkpoint identity."""
    completed = {"config": {"task": "transient_drying", "run": {"layout": {"trial": "0000__study"}}}}

    with pytest.raises(RuntimeError, match="trials keep no checkpoints"):
        artifact_generator._validate_transient_role_evidence(cast("Any", object()), completed=completed, role=cast("Any", object()))


def test_saved_prediction_comparison_viewers_are_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interact with saved-payload comparison controls without writing any file."""
    import ipywidgets as widgets  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from src import analysis  # noqa: PLC0415

    root = tmp_path / "saved-predictions"
    record = _cache_test_record(_CASE_ID, mode="autonomous_full")
    static = record.static_conditioning.copy()
    static[artifact.STATIC_ORDER.index("rho_bu_dry")] = 100.0
    scalars = record.scalar_conditioning.copy()
    scalars[artifact.SCALAR_ORDER.index("f_surf")] = 0.25
    identity = dict(record.identity)
    identity["spatial_representation"] = _spatial_identity(static)
    record = replace(record, static_conditioning=static, scalar_conditioning=scalars, identity=identity).validated()
    metadata = transient_evaluation.production_frame((transient_evaluation.sequence_record("autonomous_full", "full"),), artifact_root=str(root))
    provenance = _cache_test_provenance()
    # The shared fixture keys timing evidence by the case ID "case"; bind it to this artifact's package case ID.
    fixture_evaluation = json.dumps(metadata.attrs["artifact_provenance"]["evaluation"]).replace('"case"', json.dumps(_CASE_ID))
    provenance["evaluation"].update(json.loads(fixture_evaluation))
    provenance["dataset"].update(metadata.attrs["artifact_provenance"]["dataset"])
    fitted_scaling = artifact_generator.TransientScalingArtifact.from_state_dict(transient_evaluation.scaling_state())
    statistics = artifact_generator._record_metric_statistics(record, scaling=fitted_scaling)
    artifact.write_transient_sequence_artifact(
        root,
        dataset_name="transient_dataset",
        dataset_role="id",
        records=(record,),
        provenance=provenance,
        metric_statistics={record.record_id: statistics},
    )
    index = artifact.load_transient_sequence_artifact_index(root)
    index.frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "artifact_root": str(root),
            "artifact_provenance": dict(index.provenance),
            "transient_sequence_index": index,
            "transient_unavailable_horizons": index.unavailable_horizons,
            "transient_scaling_state": transient_evaluation.scaling_state(),
        }
    )
    evaluation = transient_session.TransientEvaluationSession({"Saved model A ID": index.frame, "Saved model B ID": index.frame})
    before = {path.relative_to(tmp_path): common.serialization.file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    analysis.evaluation.panel.build_transient_panel(session=evaluation, comparison=True, sections=("runtime", "matched_stages"))
    monkeypatch.setattr(common.paths, "get_storage_root", lambda: tmp_path)
    export_state: dict[str, Any] = {"fig": None}
    viewer = analysis.evaluation.panel._make_transient_runtime_viewer(session=evaluation, kind="tradeoff", export_state=export_state)
    assert isinstance(viewer, widgets.Box)
    selector = viewer.children[0]
    assert isinstance(selector, widgets.Dropdown)
    for _label, value in selector.options:
        selector.value = value
        assert export_state["fig"] is not None
        assert len(export_state["fig"].axes[0].collections) == 2
    sample = analysis.evaluation.panel._make_transient_sample_viewer(
        session=evaluation,
        fields=analysis.evaluation.panel._transient_channel_fields(evaluation),
        comparison=True,
        export_state=export_state,
        export_plot_name="saved_sample",
        export_title="Saved prediction comparison",
    )
    assert isinstance(sample, widgets.VBox)
    assert index.cache_size == 1
    after = {path.relative_to(tmp_path): common.serialization.file_sha256(path) for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before  # Interacting neither writes figures nor changes scientific inputs.
    evaluation.close()
    plt.close("all")
