"""Bind a Dataset to the complete candidate outcomes of its Source.

Responsibilities:
  - Check that a Dataset manifest records the replacement policy, every candidate outcome and the original-candidate failure counts
  - Check that the manifest validator rejects counts or selected cases that disagree with the recorded table
  - Check the readable failure-rate summary of the Dataset inspection

Design principles:
  - A fake-solver Source with one failed original and one pre-registered replacement exercises the real publication chain
  - Selected successes alone never describe the population they were drawn from

This module does NOT:
  - Judge the scientific validity of replacement or of the failed candidates
  - Exercise Dataset-plan orchestration, which has its own tests
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from support.generation_fixtures import successful_source_case

from src import common, datasets, generation
from src.datasets import dataset_packages as facade
from src.datasets.packages import dataset_packages_manifest as manifests
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.publication import generation_publication_campaign_evidence as campaign_evidence
from src.generation.publication import generation_publication_catalog as catalog

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

_SELECTOR_KEYS = (
    "campaign_name",
    "campaign_digest",
    "material_family",
    "case_id",
    "case_input_id",
    "simulation_case_id",
    "case_hdf5_sha256",
)


@pytest.mark.usefixtures("_dataset_producer_evidence")
def test_dataset_manifest_records_replacement_policy_and_every_candidate_outcome(
    generation_config_factory: Any,
    fake_comsol: Path,
    tmp_path: Path,
) -> None:
    """One failed original, one pre-registered replacement: the Dataset states the failure count and the linkage."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", executable=fake_comsol, natural_count=2)
    authored = {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "replacement_capacity": {"lentil": 1}}
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=authored)
    storage = tmp_path / "scientific-storage"
    lifecycle.create_campaign(campaign, storage_root=storage)
    primary = campaign.batches[0]
    inputs = generation.cases.input_generation.generate_input_cases(primary, 2, storage_root=storage)
    first = generation.cases.input_generation.admit_configured_input_references(primary, storage_root=storage)[1].case_payload()
    lifecycle.decide_candidate(
        "steady_flow",
        campaign.source_name,
        "lentil",
        "case_0001",
        outcome="failed",
        stage="solver",
        failure_class="solver",
        reason_code="synthetic_nonconvergence",
        detail="Synthetic non-converging solve",
        case_input_id=first["case_input_id"],
        simulation_case_id=first["simulation_case_id"],
        case_hdf5_relative=None,
        case_hdf5_sha256=None,
        evidence={"input_manifest_relative": (inputs.metadata_directory / "input_generation_manifest.json").relative_to(storage).as_posix()},
        storage_root=storage,
    )
    successful_source_case(campaign.source_name, primary, 2, storage, tmp_path / "work")
    replacement = lifecycle.materialize_replacement_batch(campaign, primary.batch_id, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(replacement, 1, storage_root=storage)
    successful_source_case(campaign.source_name, replacement, 3, storage, tmp_path / "work")
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            primary.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(primary.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), run_manifest)

    rows = catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)
    request = {
        "schema_kind": "dataset_selection_request",
        "schema_version": 2,
        "source_task": "steady_flow",
        "dataset_name": "outcomes",
        "dataset_view": "steady_flow",
        "selection": {"kind": "exact"},
        "source_cases": [{key: row[key] for key in _SELECTOR_KEYS} for row in rows if row["outcome"] == "successful"],
    }
    published = datasets.packages.build_dataset_selection(request, storage_root=storage)
    manifest = datasets.packages.load_package_manifest("steady_flow", published["dataset_id"], storage_root=storage)

    assert manifest["schema_version"] == manifests.DATASET_PACKAGE_SCHEMA_VERSION
    (outcome,) = manifest["source_outcomes"]
    assert outcome["source_name"] == campaign.source_name
    assert outcome["materials"] == {
        "lentil": {
            "original_candidates": 2,
            "original_failed": 1,
            "replacement_candidates": 1,
            "replacement_failed": 0,
            "successful": 2,
            "pending": 0,
        }
    }
    (policy,) = outcome["replacement_policy"]
    assert (policy["material_family"], policy["planned_capacity"], policy["high_water"]) == ("lentil", 1, 1)
    assert [(item["registered_by"], item["high_water"]) for item in policy["amendments"]] == [("plan", 1)]
    table = {row["case_id"]: row for row in outcome["candidates"]}
    assert list(table) == ["case_0001", "case_0002", "case_0003"]
    assert (table["case_0001"]["origin"], table["case_0001"]["outcome"], table["case_0001"]["reason_code"]) == (
        "primary",
        "failed",
        "synthetic_nonconvergence",
    )
    assert (table["case_0003"]["origin"], table["case_0003"]["target_slot"], table["case_0003"]["parent_failed_case_id"]) == (
        "replacement",
        1,
        "case_0001",
    )
    assert table["case_0002"]["parent_failed_case_id"] is None
    assert [row["selected"] for row in outcome["candidates"]] == [False, True, True]

    summary = "\n".join(facade._source_outcome_lines(manifest["source_outcomes"]))
    assert f"Source outcomes of {campaign.source_name}" in summary
    assert "lentil: original candidates failed 1/2 (50.0%); replacements used 1 (failed 0) of capacity 1 (pre-registered 1); successful 2" in summary

    payload = common.paths.resolve_dataset_path(
        "steady_flow", published["dataset_id"], dataset_root=common.paths.get_datasets_root(storage_root=storage)
    )
    for mutate, message in (
        (lambda value: value["source_outcomes"][0]["materials"]["lentil"].update(original_failed=0), "counts of 'lentil' disagree"),
        (lambda value: value["source_outcomes"][0]["candidates"][1].update(outcome="failed"), r"every selected case as a successful candidate"),
        (lambda value: value["source_outcomes"][0].update(source_name="another-source"), "exactly the contributing Sources"),
    ):
        tampered = copy.deepcopy(manifest)
        mutate(tampered)
        with pytest.raises(ValueError, match=message):
            manifests.validate_manifest_content(
                tampered, task="steady_flow", dataset_id=published["dataset_id"], payload_path=payload, validate_payload_hash=False
            )
