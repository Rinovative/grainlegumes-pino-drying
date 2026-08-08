"""
Build the unregistered synthetic TaskSpec used by generic contract tests.

The test-only three-input/two-output contract uses arbitrary neutral field names,
units, metrics, and dataset IDs to prove current TaskSpec consumers do not hardcode
the steady-flow channel contract. It is not a production task implementation and
is never registered, trained, or published.
"""

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from src import domain


def build_synthetic_generated_batch_identity(
    *,
    batch_id: str,
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Return one deterministic current profile-qualified batch identity."""
    case_ids = list(sample_ids)
    content: dict[str, Any] = {
        "schema_version": 2,
        "batch_id": batch_id,
        "simulation_profile": "steady_flow",
        "batch_identity": hashlib.sha256(f"{batch_id}:batch".encode()).hexdigest(),
        "scientific_config_digest": hashlib.sha256(f"{batch_id}:scientific".encode()).hexdigest(),
        "template": {
            "relative_path": "simulation/steady_flow/template_brinkman.mph",
            "sha256": hashlib.sha256(b"synthetic-template").hexdigest(),
        },
        "export_contract_sha256": hashlib.sha256(b"synthetic-exports").hexdigest(),
        "available_learning_views": ["steady_flow"],
        "airflow_source": "comsol_steady_reference",
        "intended_case_ids": case_ids,
        "cases": [
            {
                "case_id": case_id,
                "material_family": "lentil",
                "case_input_id": hashlib.sha256(f"{batch_id}:{case_id}:input".encode()).hexdigest(),
                "simulation_case_id": hashlib.sha256(f"{batch_id}:{case_id}:simulation".encode()).hexdigest(),
                "case_hdf5_sha256": hashlib.sha256(f"{batch_id}:{case_id}:hdf5".encode()).hexdigest(),
                "success_sha256": hashlib.sha256(f"{batch_id}:{case_id}:success".encode()).hexdigest(),
                "provenance_sha256": hashlib.sha256(f"{batch_id}:{case_id}:provenance".encode()).hexdigest(),
            }
            for case_id in case_ids
        ],
    }
    encoded = json.dumps(
        content,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    content["batch_manifest_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return content


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
        default_datasets=spec.DatasetDefaults(train="synthetic_train", ood=("synthetic_ood",)),
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
            allowed_continuities=("none",),
            boundary="none",
        ),
    )
