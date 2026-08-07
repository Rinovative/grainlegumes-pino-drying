# ruff: noqa: S101
"""Current-profile simulation-to-steady-flow dataset integration contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from src import datasets, domain, generation

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("simulation_profile", "airflow_source", "views"),
    [
        ("steady_flow", "comsol_steady_reference", ["steady_flow"]),
        (
            "transient_drying",
            "comsol_coupled_reference",
            ["steady_flow", "transient_drying"],
        ),
    ],
)
def test_both_profiles_publish_the_same_steady_flow_dataset_contract(
    simulation_profile: str,
    airflow_source: str,
    views: list[str],
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """Build a current steady-flow dataset from each profile's airflow view."""
    config_path, _template = generation_config_factory(
        simulation_profile=simulation_profile,
        executable=fake_comsol,
    )
    config = generation.config.load_generation_config(config_path)
    storage = tmp_path / f"storage-{simulation_profile}"
    generation.runtime.run_case(
        config,
        1,
        cores_per_case=1,
        storage_root=storage,
        work_root=tmp_path / f"work-{simulation_profile}",
    )
    generation.runtime.finalize_batch(config, storage_root=storage)

    dataset_id = f"{simulation_profile}_airflow"
    result = datasets.build.build_batch_dataset(
        config.batch_id,
        dataset_id=dataset_id,
        storage_root=storage,
    )
    payload = torch.load(result["dataset_path"], map_location="cpu", weights_only=False)
    task = domain.tasks.registry.get_task("steady_flow")
    identity = datasets.identity.validate_training_dataset_payload(
        payload,
        task=task,
        verify_content=True,
    )
    package = datasets.metadata.load_dataset_metadata(
        dataset_id,
        dataset_identity=identity,
        metadata_root=storage / "02_datasets" / "meta",
        dataset_path=result["dataset_path"],
    )

    assert payload["task"] == "steady_flow"
    assert payload["fields"] == {
        "inputs": list(task.input_names),
        "outputs": list(task.output_names),
    }
    assert payload["inputs"].shape[1] == task.in_channels
    assert payload["outputs"].shape[1] == task.out_channels
    assert payload["source_provenance"]["simulation_profile"] == simulation_profile
    assert payload["source_provenance"]["airflow_source"] == airflow_source
    assert payload["source_provenance"]["available_learning_views"] == views
    assert package.source_manifest["simulation_profile"] == simulation_profile
    assert package.source_manifest["template"]["sha256"] == config.template_sha256
    assert payload["source_provenance"]["batch_manifest_sha256"] == package.source_manifest_sha256
    assert result["source_profile"] == simulation_profile
    assert package.timing_summary["status"] == "unavailable"
