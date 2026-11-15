"""Provide compact synthetic transient Evaluation records, frames, and scaling state.

Responsibilities:
  - Build in-memory sequence records and artifact frames with complete provenance and timing

Design principles:
  - Values are small, exact, and test-owned so session and artifact tests share one fixture shape
  - States are constant over the smallest grid with an H1 interior, so per-case errors equal the offsets

This module does NOT:
  - Run inference, write artifacts, or represent production results
"""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_transient_timing as timing
from support.scaling import spatial_view, transient_scaling

ANALYSIS_SHA256 = "9" * 64
GRID_SHAPE = (5, 5)
_STATIC_CONSTANTS = (1.0, 1.0, 101325.0, 0.4, 100.0)


def scaling_state(*, membership_digest: str = "a" * 64) -> dict[str, object]:
    """Return one valid synthetic transient scaling state with a two-unit state scale and unit scales elsewhere."""
    view = spatial_view(GRID_SHAPE)
    return transient_scaling(view, state_std=2.0, membership_digest=membership_digest, dataset_identity={"synthetic": "session"}).state_dict()


def static_conditioning() -> np.ndarray:
    """Return static fields on a 4 cm by 8 cm grid with constant u, v, p, porosity and dry bulk density."""
    y, x = np.meshgrid(np.linspace(0.0, 0.04, GRID_SHAPE[0]), np.linspace(0.0, 0.08, GRID_SHAPE[1]), indexing="ij")
    return np.stack((x, y, *(np.full(GRID_SHAPE, value) for value in _STATIC_CONSTANTS))).astype(np.float32)


def sequence_record(
    mode: str,
    horizon: int | str,
    *,
    case_id: str = "case",
    offset: float = 2.0,
    material_family: str = "lentil",
    length: int | None = None,
) -> SimpleNamespace:
    """Return one synthetic four-channel sequence record with a constant prediction offset after the stored origin."""
    if length is None:
        length = 2 if horizon == 1 else 3
    reference = np.zeros((length, 4, *GRID_SHAPE), dtype=np.float32)
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
        static_conditioning=static_conditioning(),
        scalar_conditioning=np.asarray([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]),
        identity={
            "material_family": material_family,
            "input_profile": "canonical_physics_complete",
            "model_kind": "fno",
            "pt_payload_identity": "a" * 64,
            "timing_evidence_identity": "b" * 64,
            "scaling_identity": {"semantic_digest": "c" * 64},
            "evaluation_assignment": None,
            "spatial_representation": {"grid_identity_sha256": "f" * 64},
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
    """Return a sequence frame with complete run, Dataset, timing, and lineage provenance and the completed run's summary."""
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
                solver_hostname="solver-node",
                solver_requested_cores=8,
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
                        "planned": None,
                        "actual": {"arm": "B", "stage": "stage_b_self_fed"},
                        "training_outcome": {"status": "completed", "budget_progress": 1.0, "eligible_points": 1, "divergence": None},
                        "budget": {
                            "unit": "train_seconds",
                            "total": 1.0,
                            "units": 4,
                            "train_seconds": 1.0,
                            "validated_point": 20,
                            "skipped_points": [],
                        },
                    },
                },
            },
            # The loader attaches the completed run's summary; this one records only the run and its arm.
            dataframe.COMPLETED_RUN_SUMMARY_ATTR: {
                "task": "transient_drying",
                "run_name": "transient-fixture",
                "ended_at": "2026-11-11T00:00:00+00:00",
                "terminal_controller": {"arm": "B"},
            },
        }
    )
    return frame
