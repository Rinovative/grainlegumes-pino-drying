"""Build synthetic task declarations and exact test-owned selection provenance.

Responsibilities:
  - Construct an unregistered three-input/two-output TaskSpec and steady-source provenance

Design principles:
  - Neutral fields expose hardcoded task assumptions; provenance uses deterministic case-bound
    digests

This module does NOT:
  - Register a production task, train it, or publish scientific packages
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src import common, domain
from src.datasets.contracts import dataset_contracts_identity as dataset_identity
from src.datasets.contracts import dataset_contracts_metadata as dataset_metadata
from src.datasets.contracts import dataset_contracts_partition as partition_contract
from src.datasets.contracts import dataset_contracts_views as views
from src.datasets.packages import dataset_packages_reproduction as reproduction


def build_synthetic_selection_provenance(
    *,
    dataset_name: str,
    sample_ids: Sequence[str],
    material_family: str = "lentil",
    material_role: str | None = None,
) -> dict[str, Any]:
    """
    Return current Dataset provenance for bounded synthetic cases.

    ``material_role`` ``"id"`` assigns a fixed Train/validation/ID-test partition
    (two, one and one cases for four cases), ``"ood"`` holds every case out, and
    ``None`` leaves the selection role-neutral.
    """
    task = domain.tasks.registry.get_task("steady_flow")
    view = views.get_view("steady_flow")

    def digest(label: str) -> str:
        return hashlib.sha256(f"{dataset_name}:{label}".encode()).hexdigest()

    cases = []
    for position, case_id in enumerate(sample_ids):
        cases.append(
            {
                "package_case_id": case_id,
                "source": {
                    "task": "steady_flow",
                    "campaign_name": f"{dataset_name}_synthetic_source",
                    "source_name": f"{dataset_name}_synthetic_source",
                    "campaign_digest": digest("campaign"),
                    "source_plan_sha256": digest("plan"),
                    "material_family": material_family,
                    "domain_profile": "reference_range",
                    "source_batch_id": "synthetic_batch",
                    "case_id": case_id,
                    "case_input_id": digest(f"{case_id}:input"),
                    "simulation_case_id": digest(f"{case_id}:simulation"),
                    "case_hdf5_relative": f"{case_id}/case.h5",
                    "case_hdf5_sha256": digest(f"{case_id}:hdf5"),
                    "reservation_sha256": digest(f"{case_id}:reservation"),
                    "decision_sha256": digest(f"{case_id}:decision"),
                    "selected_pool_sha256": None,
                },
                "design": {
                    "sampling_method": "lhs",
                    "original_design_size": len(sample_ids),
                    "candidate_position": position,
                    "origin": "primary",
                    "root_seed": 7,
                    "candidate_seed": position + 1,
                    "derived_seeds": {},
                    "design_evidence": {},
                },
            }
        )
    return {
        "schema_kind": "dataset_selection_manifest",
        "schema_version": 3,
        "dataset_name": dataset_name,
        "dataset_view": "steady_flow",
        "registered_task_id": task.id,
        "source_task": "steady_flow",
        "materials": [material_family],
        "channel_contract": {
            "input": [field.as_dict() for field in task.inputs],
            "target": [field.as_dict() for field in task.outputs],
            "tensor_layout": list(task.tensor_layout),
        },
        "channel_contract_digest": view.contract_digest,
        "selection": {"kind": "exact"},
        "case_partition": _synthetic_partition(sample_ids, material_family=material_family, material_role=material_role),
        "plan_invocation": None,
        "grid": {"shape": [8, 8], "axes_sha256": digest("grid")},
        "source_case_identities": cases,
        "source_outcomes": [
            {
                "source_name": f"{dataset_name}_synthetic_source",
                "campaign_digest": digest("campaign"),
                "replacement_policy": [],
                "materials": {
                    material_family: {
                        "original_candidates": len(sample_ids),
                        "original_failed": 0,
                        "replacement_candidates": 0,
                        "replacement_failed": 0,
                        "successful": len(sample_ids),
                        "pending": 0,
                    }
                },
                "candidates": [
                    {
                        "case_id": case_id,
                        "material_family": material_family,
                        "sampling_regime": "natural",
                        "origin": "primary",
                        "candidate_position": position,
                        "target_slot": position,
                        "parent_failed_case_id": None,
                        "outcome": "successful",
                        "selected": True,
                        "failure_stage": None,
                        "failure_class": None,
                        "reason_code": None,
                    }
                    for position, case_id in enumerate(sample_ids)
                ],
            }
        ],
        "builder_identity": {
            **dataset_identity.dataset_conversion_contract_identity("steady_flow"),
            "implementation_sha256": reproduction.conversion_implementation_sha256("steady_flow"),
        },
        "producer_evidence": {
            "source_snapshot": {
                "relative_path": (f"source_snapshots/{'a' * 40}-{'b' * 64}/snapshot.json"),
                "sha256": digest("snapshot"),
                "git_commit": "a" * 40,
                "source_fingerprint_sha256": "b" * 64,
            },
            "environment": {
                "python": "test-fixture",
                "platform": "test-fixture",
                "packages": {},
            },
        },
    }


def _synthetic_partition(
    sample_ids: Sequence[str],
    *,
    material_family: str,
    material_role: str | None,
) -> dict[str, Any] | None:
    """Build one fixed synthetic Dataset partition through the production contract."""
    if material_role is None:
        return None
    case_ids = list(sample_ids)
    half = max(1, len(case_ids) // 2)
    recipe = {
        "kind": "exact",
        "train": case_ids[:half],
        "validation": case_ids[half : half + 1],
        "id_test": case_ids[half + 1 :],
    }
    return partition_contract.build_fixed_partition(
        case_ids,
        material_by_case=dict.fromkeys(case_ids, material_family),
        material_roles={material_family: material_role},
        recipe=recipe,
        seed_input=9,
        seed=common.random.derive_subseed(9, "split"),
    )


def build_synthetic_task() -> domain.tasks.spec.TaskSpec:
    """
    Return a coherent task with distinct fields, per-field metrics, and no physics.

    Returns
    -------
    domain.tasks.spec.TaskSpec
        Unregistered two-output contract used to prove consumers derive channel
        counts, names, units, and metric definitions from TaskSpec.

    Notes
    -----
    The fixture intentionally simplifies production assumptions: fields use
    arbitrary units, preprocessing uses generic standardization, dataset IDs are
    non-executable labels, and physics kind/equations/boundary are ``none``.

    """
    spec = domain.tasks.spec
    return spec.TaskSpec(
        id="synthetic_field_task",
        schema_version=spec.TASK_SCHEMA_VERSION,
        inputs=(
            spec.FieldSpec("feature_a", "state", "unit_in_a", "identity"),
            spec.FieldSpec("feature_b", "state", "unit_in_b", "identity"),
            spec.FieldSpec("feature_c", "state", "unit_in_c", "identity"),
        ),
        outputs=(
            spec.FieldSpec("response_a", "state", "unit_out_a", "identity"),
            spec.FieldSpec("response_b", "state", "unit_out_b", "identity"),
        ),
        output_groups=(
            spec.OutputGroupSpec("quantity_a", ("response_a",)),
            spec.OutputGroupSpec("quantity_b", ("response_b",)),
        ),
        tensor_layout=("batch", "channel", "y", "x"),
        operator_axes=(2, 3),
        normalization_axes=(0, 2, 3),
        preprocessing=spec.PreprocessingSpec(
            input_normalization="standard",
            output_normalization="standard",
            fit_split="train",
        ),
        data_losses=("relative_l2",),
        default_metrics=(
            spec.MetricSpec(
                id="normalized_group_macro_rmse",
                kind="group_macro_rmse",
                space="physical",
                fields=("response_a", "response_b"),
                reduction="group_macro_element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="normalized_relative_l2",
                kind="relative_l2",
                space="normalized",
                fields=("response_a", "response_b"),
                reduction="sample_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_a",
                kind="rmse",
                space="physical",
                fields=("response_a",),
                reduction="element_mean",
                direction="minimize",
            ),
            spec.MetricSpec(
                id="physical_rmse_response_b",
                kind="rmse",
                space="physical",
                fields=("response_b",),
                reduction="element_mean",
                direction="minimize",
            ),
        ),
        physics=spec.PhysicsSpec(
            kind="none",
            equation_set="none",
            continuity="none",
            boundary="none",
        ),
    )


def save_steady_package(dataset_root: Path, payload: dict[str, Any]) -> Path:
    """Publish one bounded synthetic steady payload with its current Dataset manifest."""
    dataset_id = str(payload["dataset_id"])
    task = domain.tasks.registry.get_task(str(payload["task"]))
    metadata_dir = common.paths.resolve_dataset_dir("steady_flow", dataset_id, dataset_root=dataset_root)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    destination = common.paths.resolve_dataset_path("steady_flow", dataset_id, dataset_root=dataset_root)
    common.serialization.atomic_torch_save(payload, destination)
    identity = dataset_identity.validate_training_dataset_payload(payload, task=task, verify_content=True)
    provenance = payload["source_provenance"]
    resolved_id, resolved_digest = dataset_identity.package_identity_from_provenance(provenance)
    if resolved_id != dataset_id or identity.source_selection_identity is None:
        msg = "Synthetic steady payload does not bind its selection provenance."
        raise ValueError(msg)
    manifest = {
        **provenance,
        "dataset_id": resolved_id,
        "dataset_digest": resolved_digest,
        "payload_filename": destination.name,
        "source_case_count": len(provenance["source_case_identities"]),
        "sample_count": identity.sample_count,
        "transition_count": 0,
        "payload_sha256": common.serialization.file_sha256(destination),
    }
    common.serialization.atomic_write_json(metadata_dir / common.paths.DATASET_MANIFEST_FILENAME, manifest)
    dataset_metadata.validate_dataset_metadata_directory(metadata_dir, dataset_identity=identity, dataset_path=destination)
    return destination
