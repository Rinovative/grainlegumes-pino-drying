"""Provide compact synthetic transient Evaluation records, frames, and scaling state.

Responsibilities:
  - Build in-memory sequence records and artifact frames with complete provenance and timing

Design principles:
  - Values are small, exact, and test-owned so session and artifact tests share one fixture shape

This module does NOT:
  - Run inference, write artifacts, or represent production results
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src import datasets, domain
from src.analysis.evaluation import evaluation_transient_timing as timing
from src.learning.transient.learning_transient_contracts import TemporalConditioningSpec, TransientTensorizerSpec
from src.learning.transient.learning_transient_scaling import SCALE_FLOOR, TransientScalingArtifact

ANALYSIS_SHA256 = "9" * 64


def scaling_state() -> dict[str, object]:
    """Return one valid synthetic transient scaling state with a two-unit state scale."""
    spec = TransientTensorizerSpec("canonical_physics_complete", TemporalConditioningSpec("none"))
    contract = datasets.contracts.transient.TRANSIENT_STEP_CONTRACT
    names = tuple(
        tuple(field.name for field in group)
        for group in (contract.dynamic_state, contract.static_spatial_conditioning, contract.step_boundary_conditioning, contract.scalar_conditioning)
    )
    artifact = TransientScalingArtifact(
        task_contract_digest=domain.tasks.registry.get_task("transient_drying").contract_digest,
        data_contract_digest=domain.tasks.registry.get_task("transient_drying").data_contract_digest,
        tensorizer=spec,
        dataset_identity={"synthetic": "session"},
        train_membership_digest="a" * 64,
        scale_mode="state_std",
        numerical_floor=SCALE_FLOOR,
        unique_train_state_count=1,
        unique_transition_count=1,
        transition_count=1,
        spatial_shape=(1, 1),
        state_names=names[0],
        static_names=names[1],
        boundary_names=names[2],
        scalar_names=names[3],
        state_mean=torch.zeros(4),
        state_std=torch.full((4,), 2.0),
        delta_rms=torch.ones(4),
        increment_scale=torch.full((4,), 2.0),
        static_mean=torch.zeros(7),
        static_std=torch.ones(7),
        scalar_mean=torch.zeros(8),
        scalar_std=torch.ones(8),
        omega_boundary_mean=torch.tensor(0.0),
        omega_boundary_std=torch.tensor(1.0),
        horizon=8.0,
    )
    return artifact.state_dict()


def sequence_record(
    mode: str,
    horizon: int | str,
    *,
    case_id: str = "case",
    offset: float = 2.0,
    material_family: str = "lentil",
) -> SimpleNamespace:
    """Return one synthetic four-channel sequence record with a constant prediction offset."""
    length = 2 if horizon == 1 else 3
    reference = np.zeros((length, 4, 1, 1), dtype=np.float32)
    prediction = reference.copy()
    prediction[1:] = offset
    full_reference = mode == "autonomous_full" and horizon == "full"
    target = {
        "predicted_evidence_scope": "regular_sequence_grid",
        "predicted_available": True,
        "predicted_unavailable_reason": None,
        "predicted_reached": True,
        "predicted_time_to_target": 1.0,
        "predicted_final_gap": -0.1,
        "predicted_final_time": float(length - 1),
        "reference_evidence_scope": ("canonical_completed_case" if full_reference else "unavailable_partial_interval"),
        "reference_available": full_reference,
        "reference_unavailable_reason": (None if full_reference else "canonical_completed_case_target_unavailable_for_partial_interval"),
        "reference_reached": False,
        "reference_time_to_target": None,
        "reference_final_gap": (0.2 if full_reference else None),
        "reference_final_time": (float(length - 1) if full_reference else None),
    }
    return SimpleNamespace(
        mode=mode,
        case_id=case_id,
        requested_horizon=horizon,
        origin_index=0,
        physical_times=np.arange(length, dtype=np.float64),
        reference_states=reference,
        predicted_states=prediction,
        spatial_mask=np.ones((1, 1), dtype=bool),
        static_conditioning=np.asarray(
            [
                [[0.0]],
                [[0.0]],
                [[1.0]],
                [[1.0]],
                [[101325.0]],
                [[0.4]],
                [[100.0]],
            ],
            dtype=np.float32,
        ),
        scalar_conditioning=np.asarray([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
        identity={
            "material_family": material_family,
            "input_profile": "canonical_physics_complete",
            "model_kind": "fno",
            "pt_payload_identity": "a" * 64,
            "timing_evidence_identity": "b" * 64,
            "scaling_identity": {"semantic_digest": "c" * 64},
            "evaluation_assignment": None,
        },
        target=target,
    )


def sequence_frame(records: tuple[SimpleNamespace, ...]) -> pd.DataFrame:
    """Return an artifact frame carrying the given in-memory sequence records and their recorded analysis producer."""
    frame = pd.DataFrame()
    frame.attrs.update(
        {
            "artifact_kind": "transient_sequence",
            "artifact_provenance": {"producer": {"analysis_sha256": ANALYSIS_SHA256}},
            "transient_sequence_records": records,
            "transient_unavailable_horizons": ({"case_id": "case", "requested_horizon": 4, "reason": "short"},),
            "transient_scaling_state": scaling_state(),
        }
    )
    return frame


def production_frame(
    records: tuple[SimpleNamespace, ...],
    *,
    artifact_root: str,
    dataset_name: str = "transient_dataset",
    dataset_role: str = "id",
) -> pd.DataFrame:
    """Return a sequence frame with complete run, Dataset, timing, and lineage provenance."""
    frame = sequence_frame(records)
    report = timing.build_transient_timing_report(
        (
            timing.TransientTimingCase(
                case_id="case",
                repetitions={
                    "drying_no_rollout_model_seconds": (0.01, 0.02, 0.03),
                    "drying_no_prepared_rollout_wall_seconds": (0.02, 0.03, 0.04),
                    "comsol_transient_drying_seconds": (0.3,),
                },
                device="cpu",
                precision="float32",
                pt_payload_identity="a" * 64,
                warmup_passes=1,
                cpu="test-cpu",
                software_versions={"python": "test", "pytorch": "test", "numpy": "test"},
                cold_timings={"drying_no_rollout_model_seconds": 0.05},
            ),
        )
    )
    frame.attrs.update(
        {
            "artifact_root": artifact_root,
            "artifact_provenance": {
                "producer": {"analysis_sha256": ANALYSIS_SHA256},
                "run": {
                    "best_checkpoint_sha256": "d" * 64,
                    "best_checkpoint_epoch": 3,
                },
                "dataset": {
                    "name": dataset_name,
                    "role": dataset_role,
                    "source_dataset_ids": [dataset_name],
                    "source_identities": [{"dataset_id": dataset_name}],
                    "case_ids_by_source": [[record.case_id for record in records]],
                    "membership_digests": ["e" * 64],
                    "evaluation_assignments": [],
                },
                "evaluation": {
                    "pipeline_analysis": {
                        "conditions": [
                            "A_comsol_reference",
                            "B_drying_on_comsol_airflow",
                            "C_drying_on_airflow_no",
                        ],
                        "c_available": False,
                        "c_unavailable_reasons": ["compatible_airflow_checkpoint_not_selected"],
                        "fabricated_prediction_count": 0,
                    },
                },
                "timing": {
                    "report": json.loads(json.dumps(asdict(report))),
                    "component_availability": {
                        "case": {"dataset_materialization_seconds": 0.005},
                    },
                },
                "lineage": {
                    "stage_identity": {"stage": "b", "comparison_arm": "b"},
                    "training_strategy": "rollout",
                    "matched_compute_manifest": {
                        "planned": {"clock": "cuda_device_seconds"},
                        "actual": {"budget_complete": True},
                    },
                },
            },
        }
    )
    return frame
