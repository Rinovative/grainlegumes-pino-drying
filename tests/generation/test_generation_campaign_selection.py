"""Current source design, input admission, and catalog contracts.

Responsibilities:
  - Check complete candidate publication, idempotent reuse, and frozen-input admission
  - Check that an authored replacement capacity is pre-registered once, reproduced exactly on rerun and distinguishable from explicit amendments

Design principles:
  - Exact candidate bytes and frozen science govern Source admission
  - Replacement capacity is policy: it never changes the Source identity or the sampling seeds

This module does NOT:
  - Execute candidates or build Dataset packages
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from src import common, generation
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.contracts import generation_contracts_source as source
from src.generation.publication import generation_publication_catalog as catalog


def test_current_source_inputs_require_complete_plan_and_publish_exact_candidates(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Plan the full design before generating cases, without starting an external process."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = tmp_path / "scientific-storage"
    with pytest.raises(FileNotFoundError, match=r"campaign_plan\.json"):
        generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)

    root = lifecycle.create_campaign(campaign, storage_root=storage)
    with (
        patch("subprocess.Popen", side_effect=AssertionError("input generation started a process")),
        patch("subprocess.run", side_effect=AssertionError("input generation started a process")),
    ):
        inputs = generation.cases.input_generation.generate_input_cases(batch, 2, storage_root=storage)
    assert inputs.metadata_directory == root / "lentil" / "_record" / "natural"
    assert inputs.raw_directory == root / "lentil"
    assert inputs.generated_case_count == 2
    assert not list(root.rglob("input_generations"))
    references = generation.cases.input_generation.admit_configured_input_references(batch, storage_root=storage)
    assert sorted(references) == [1, 2]
    assert [references[index].case_payload()["case_id"] for index in (1, 2)] == ["case_0001", "case_0002"]
    plan = lifecycle.load_campaign_plan("steady_flow", campaign.source_name, storage_root=storage)
    assert plan["original_design_sizes"] == {"lentil": 2}
    rows = catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)
    assert [(row["case_id"], row["outcome"]) for row in rows] == [("case_0001", "pending"), ("case_0002", "pending")]


def test_source_input_republication_is_idempotent_and_preserves_case_bytes(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Reuse exact published inputs without changing the original design."""
    config_path, _template = generation_config_factory(simulation_profile="transient_drying", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    first = generation.cases.input_generation.generate_input_cases(batch, 2, storage_root=storage)
    first_bytes = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    lifecycle.create_campaign(campaign, storage_root=storage)
    second = generation.cases.input_generation.generate_input_cases(batch, 2, storage_root=storage)
    assert second.generated_case_count == 0
    assert second.reused_case_count == 2
    assert first.input_generation_id == second.input_generation_id
    assert {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()} == first_bytes


def test_input_admission_rejects_tampered_candidate_and_frozen_science(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Fail closed on changed case identity or a conflicting same-name campaign."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = tmp_path / "scientific-storage"
    lifecycle.create_campaign(campaign, storage_root=storage)
    generation.cases.input_generation.generate_input_cases(batch, 2, storage_root=storage)
    references = generation.cases.input_generation.admit_configured_input_references(batch, storage_root=storage)
    case_path = references[1].case_directory / "case.json"
    original = json.loads(case_path.read_text(encoding="utf-8"))
    corrupted = {**original, "case_input_id": "0" * 64}
    common.serialization.atomic_write_json(case_path, corrupted)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        generation.cases.input_generation.admit_configured_input_references(batch, storage_root=storage)
    common.serialization.atomic_write_json(case_path, original)
    assert generation.cases.input_generation.admit_configured_input_references(batch, storage_root=storage)[1].case_payload() == original
    unexpected = references[1].case_directory / "solved.mph"
    unexpected.write_bytes(b"unexpected execution artifact")
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        generation.cases.input_generation.admit_configured_input_references(batch, storage_root=storage)
    unexpected.unlink()


def test_producer_code_invalidates_reuse_but_documentation_preserves_seeds(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind cache and Source identity to producing code without reseeding science."""
    executing_source = Path(source.__file__).resolve().parents[2]
    recovered = tmp_path / "producer-source"
    for relative in source.GENERATION_SOURCE_PROJECTION.members(executing_source):
        destination = recovered / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(executing_source / relative, destination)
    identity = source.input_producer_identity
    monkeypatch.setattr(source, "input_producer_identity", lambda _root=None: identity(recovered))
    source_identity = source.source_producer_identity
    monkeypatch.setattr(source, "source_producer_identity", lambda: source_identity(recovered))
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    original = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "science"
    lifecycle.create_campaign(original, storage_root=storage)
    first = generation.cases.input_generation.generate_input_cases(original.batches[0], 2, storage_root=storage)
    producing_file = recovered / "generation/cases/generation_cases_fields.py"
    producing_file.write_text(producing_file.read_text() + "\n# Documentation-only clarification.\n")
    documented = generation.cases.config.load_campaign_config(config_path)
    assert documented.producer_identity == original.producer_identity
    assert documented.source_name == original.source_name
    reused = generation.cases.input_generation.generate_input_cases(documented.batches[0], 2, storage_root=storage)
    assert reused.input_generation_id == first.input_generation_id
    assert reused.reused_case_count == 2

    # A changed numerical implementation is a real source edit, not a version knob.
    producing_file.write_text(producing_file.read_text().replace("import numpy as np", "import numpy as numpy_changed", 1))
    changed = generation.cases.config.load_campaign_config(config_path)
    assert changed.producer_identity != original.producer_identity
    assert changed.source_name == original.source_name
    assert changed.campaign_digest != original.campaign_digest
    with pytest.raises(ValueError, match="--revision LABEL"):
        lifecycle.create_campaign(changed, storage_root=storage)
    renamed = {**yaml.safe_load(config_path.read_text()), "revision": "updated-producer-source"}
    changed = generation.cases.config.load_campaign_config(config_path, authored_value=renamed)
    assert changed.campaign_name == original.campaign_name
    assert changed.source_name == f"{original.source_name}__rev-updated-producer-source"
    assert changed.batches[0].seed_base == original.batches[0].seed_base
    assert changed.batches[0].assignments == original.batches[0].assignments
    lifecycle.create_campaign(changed, storage_root=storage)
    regenerated = generation.cases.input_generation.generate_input_cases(changed.batches[0], 2, storage_root=storage)
    assert regenerated.input_generation_id != first.input_generation_id
    assert regenerated.generated_case_count == 2
    assert regenerated.reused_case_count == 0

    solver_file = recovered / "generation/runtime/generation_runtime_comsol.py"
    solver_file.write_text(solver_file.read_text().replace('COMSOL_JOB_TAG: Final = "b1"', 'COMSOL_JOB_TAG: Final = "b2"'))
    solver_changed = generation.cases.config.load_campaign_config(config_path, authored_value=renamed)
    assert solver_changed.source_name == changed.source_name
    assert solver_changed.campaign_digest != changed.campaign_digest
    with pytest.raises(ValueError, match="--revision LABEL"):
        lifecycle.create_campaign(solver_changed, storage_root=storage)
    assert solver_changed.producer_identity is not None
    assert changed.producer_identity is not None
    assert solver_changed.producer_identity["implementation_sha256"] == changed.producer_identity["implementation_sha256"]
    assert generation.cases.input_generation.configured_input_generation_id(solver_changed.batches[0]) == regenerated.input_generation_id
    assert solver_changed.batches[0].seed_base == changed.batches[0].seed_base


def _authored(config_path: Path, capacity: Any) -> dict[str, Any]:
    """Return the test campaign as an in-memory authored value that carries a replacement capacity."""
    return {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "replacement_capacity": capacity}


def test_authored_replacement_capacity_is_preregistered_once_and_reproduced_exactly(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """The plan's capacity becomes amendment one, a rerun is idempotent, a different value is refused, and explicit additions stay visible."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    plain = generation.cases.config.load_campaign_config(config_path)
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, {"lentil": 2}))
    assert (campaign.source_name, campaign.campaign_digest, campaign.root_seed) == (plain.source_name, plain.campaign_digest, plain.root_seed)
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    assert (
        lifecycle.load_campaign_plan("steady_flow", campaign.source_name, storage_root=storage)["seed_key"]
        == lifecycle._resolved_plan(plain)["seed_key"]
    )
    batch = campaign.batches[0]

    def policy() -> dict[str, Any]:
        (entry,) = lifecycle.replacement_policy("steady_flow", campaign.source_name, storage_root=storage)
        return entry

    assert (policy()["planned_capacity"], policy()["high_water"]) == (2, 2)
    assert [(item["amendment_index"], item["registered_by"]) for item in policy()["amendments"]] == [(1, "plan")]
    before = {path: path.read_bytes() for path in root.rglob("amendment_*.json")}
    assert lifecycle.create_campaign(campaign, storage_root=storage) == root
    assert {path: path.read_bytes() for path in root.rglob("amendment_*.json")} == before

    other = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, {"lentil": 3}))
    assert other.source_name == campaign.source_name
    with pytest.raises(ValueError, match=r"pre-registered per Source.*records 2.*requests 3"):
        lifecycle.create_campaign(other, storage_root=storage)
    with pytest.raises(ValueError, match=r"records 2.*requests 0"):
        lifecycle.create_campaign(plain, storage_root=storage)

    lifecycle.amend_replacement_pool(
        "steady_flow", campaign.source_name, batch.batch_id, high_water=3, reason="Later explicit addition", storage_root=storage
    )
    lifecycle.create_campaign(campaign, storage_root=storage)
    assert (policy()["planned_capacity"], policy()["high_water"]) == (2, 3)
    assert [(item["registered_by"], item["reason"]) for item in policy()["amendments"]] == [
        ("plan", "authored plan generation.replacement_capacity"),
        ("explicit", "Later explicit addition"),
    ]
    with pytest.raises(ValueError, match="cannot decrease"):
        lifecycle.amend_replacement_pool("steady_flow", campaign.source_name, batch.batch_id, high_water=2, reason="Too low", storage_root=storage)
    with pytest.raises(ValueError, match="non-empty reason"):
        lifecycle.amend_replacement_pool("steady_flow", campaign.source_name, batch.batch_id, high_water=4, reason=" ", storage_root=storage)


def test_zero_authored_capacity_records_nothing_and_explicit_amendments_do_not_conflict_with_it(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """Capacity zero is the default: no amendment is written, and an explicit one on a plain Source is accepted on rerun."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, {"lentil": 0}))
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    assert not list(root.rglob("amendment_*.json"))
    late = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, {"lentil": 2}))
    with pytest.raises(ValueError, match=r"records 0.*requests 2"):
        lifecycle.create_campaign(late, storage_root=storage)
    assert not list(root.rglob("amendment_*.json"))
    run_id = f"source~steady_flow~{campaign.source_name}"
    amended = generation.campaign.amend_replacement_capacity(
        run_id, "lentil", "natural", capacity=2, reason="Add capacity after the first failure", storage_root=storage
    )
    assert (amended["previous_capacity"], amended["capacity"]) == (0, 2)
    assert (
        generation.campaign.amend_replacement_capacity(run_id, "lentil", "natural", capacity=2, reason="Repeat", storage_root=storage)["capacity"]
        == 2
    )
    lifecycle.create_campaign(campaign, storage_root=storage)
    (entry,) = lifecycle.replacement_policy("steady_flow", campaign.source_name, storage_root=storage)
    assert (entry["planned_capacity"], entry["high_water"], [item["registered_by"] for item in entry["amendments"]]) == (0, 2, ["explicit"])
    with pytest.raises(ValueError, match="no unique original batch"):
        generation.campaign.amend_replacement_capacity(run_id, "lentil", "parameter_ood", capacity=1, reason="No such batch", storage_root=storage)
    with pytest.raises(ValueError, match="canonical v2 campaign"):
        generation.campaign.amend_replacement_capacity("legacy-run", "lentil", "natural", capacity=1, reason="No handle", storage_root=storage)


@pytest.mark.parametrize(
    ("capacity", "message"),
    [
        ({"chickpea": 1}, "outside the campaign"),
        ({"lentil": -1}, "must be an integer >= 0"),
        ({"lentil": True}, "must be an integer >= 0"),
        (3, "must be a mapping"),
    ],
)
def test_authored_replacement_capacity_is_validated_and_belongs_to_plans(
    generation_config_factory: Any,
    capacity: Any,
    message: str,
) -> None:
    """Reject unknown materials, negative or non-integer counts, and a capacity written into a standalone campaign file."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    with pytest.raises((generation.cases.config.GenerationConfigError, TypeError), match=message):
        generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, capacity))
    config_path.write_text(yaml.safe_dump(_authored(config_path, {"lentil": 1}), sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="pre-registered by a Dataset plan"):
        generation.cases.config.load_campaign_config(config_path)
