"""Current source design, input admission, and catalog contracts.

Responsibilities:
  - Check complete candidate publication, idempotent reuse, and frozen-input admission
  - Check the success-rate bound, the stop rule and the tolerated failures that size admission
  - Check that a plan's minimum success rate is pre-registered once, reproduced exactly on rerun and only lowered explicitly
  - Check that only a plan campaign without an explicit cap records its admission cap as derived (null)

Design principles:
  - Exact candidate bytes and frozen science govern Source admission
  - The minimum success rate is policy: it never changes the Source identity or the sampling seeds

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
from src.generation.cases import generation_cases_selection as selection
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


@pytest.mark.parametrize(("successful", "decided", "bound"), [(0, 1, 0.95), (1, 3, 0.8646), (3, 3, 1.0)])
def test_success_rate_bound_is_the_one_sided_clopper_pearson_upper_limit(successful: int, decided: int, bound: float) -> None:
    """Bound k of n decided candidates by the 95 % Beta quantile, and by one when every decided candidate succeeded."""
    assert selection.success_rate_upper_bound(successful, decided) == pytest.approx(bound, abs=1e-4)


@pytest.mark.parametrize(
    ("successful", "decided", "rate", "stopped"),
    [
        (0, 0, 1.0, False),  # nothing decided never stops a batch
        (0, 1, 0.9, False),  # one failure leaves the bound at 0.95
        (0, 2, 0.9, True),  # a broken batch stops after its second failure
        (1, 3, 0.8, False),  # the bound 0.865 is still at or above s
        (1, 4, 0.8, True),  # the bound 0.751 falls below s
    ],
)
def test_a_batch_is_stopped_exactly_when_its_bound_falls_below_its_minimum_success_rate(
    successful: int,
    decided: int,
    rate: float,
    stopped: bool,
) -> None:
    """Stop a batch with at least one decision whose upper bound lies below s, and never while the bound reaches s."""
    assert selection.success_rate_stopped(successful, decided, minimum_success_rate=rate) is stopped


@pytest.mark.parametrize(("count", "rate", "failures"), [(150, 0.9, 24), (30, 0.6, 30), (2, 1.0, 0)])
def test_tolerated_failures_are_the_most_a_batch_can_have_and_still_reach_its_count(count: int, rate: float, failures: int) -> None:
    """The last success is admitted at count - 1 successes and f failures; one more failure would stop the batch first."""
    assert selection.maximum_tolerated_failures(count, rate) == failures
    assert not selection.success_rate_stopped(count - 1, count - 1 + failures, minimum_success_rate=rate)
    assert selection.success_rate_stopped(count - 1, count + failures, minimum_success_rate=rate)


@pytest.mark.parametrize("rate", [1e-20, 1e-300])
def test_tolerated_failures_refuse_a_rate_too_small_for_a_bounded_cap(rate: float) -> None:
    """A rate whose tolerated failures exceed the search bound raises an actionable ValueError, never a scipy error or a hang."""
    with pytest.raises(ValueError, match="too small"):
        selection.maximum_tolerated_failures(30, rate)


def _authored(config_path: Path, rate: Any) -> dict[str, Any]:
    """Return the test campaign as an in-memory authored value that carries a Dataset plan's minimum success rate."""
    return {**yaml.safe_load(config_path.read_text(encoding="utf-8")), "minimum_success_rate": rate}


def test_a_plan_rate_is_preregistered_once_reproduced_exactly_and_only_lowered_explicitly(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """The plan's s becomes amendment one; a rerun with another s is refused; an explicit amendment lowers s and a raise is refused."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    plain = generation.cases.config.load_campaign_config(config_path)
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, 0.8))
    assert (campaign.source_name, campaign.campaign_digest, campaign.root_seed) == (plain.source_name, plain.campaign_digest, plain.root_seed)
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    batch = campaign.batches[0]

    def policy() -> dict[str, Any]:
        (entry,) = lifecycle.replacement_policy("steady_flow", campaign.source_name, storage_root=storage)
        return entry

    assert (policy()["minimum_success_rate"], policy()["successful"], policy()["decided"], policy()["stopped"]) == (0.8, 0, 0, False)
    assert [(item["amendment_index"], item["registered_by"]) for item in policy()["amendments"]] == [(1, "plan")]
    before = {path: path.read_bytes() for path in root.rglob("amendment_*.json")}
    assert lifecycle.create_campaign(campaign, storage_root=storage) == root
    assert {path: path.read_bytes() for path in root.rglob("amendment_*.json")} == before
    other = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, 0.7))
    with pytest.raises(ValueError, match=r"pre-registered per Source.*records 0\.8.*requests 0\.7"):
        lifecycle.create_campaign(other, storage_root=storage)
    with pytest.raises(ValueError, match=r"records 0\.8.*requests None"):
        lifecycle.create_campaign(plain, storage_root=storage)

    lowered = lifecycle.amend_minimum_success_rate(
        "steady_flow", campaign.source_name, batch.batch_id, minimum_success_rate=0.6, reason="Known hard corner", storage_root=storage
    )
    assert lowered == (0.8, 0.6)
    assert lifecycle.amend_minimum_success_rate(
        "steady_flow", campaign.source_name, batch.batch_id, minimum_success_rate=0.6, reason="Repeat", storage_root=storage
    ) == (0.6, 0.6)
    lifecycle.create_campaign(campaign, storage_root=storage)
    assert policy()["minimum_success_rate"] == 0.6
    assert [(item["registered_by"], item["minimum_success_rate"], item["reason"]) for item in policy()["amendments"]] == [
        ("plan", 0.8, "authored plan generation.minimum_success_rate"),
        ("explicit", 0.6, "Known hard corner"),
    ]
    with pytest.raises(ValueError, match="can only be lowered"):
        lifecycle.amend_minimum_success_rate(
            "steady_flow", campaign.source_name, batch.batch_id, minimum_success_rate=0.7, reason="Raise", storage_root=storage
        )
    with pytest.raises(ValueError, match="non-empty reason"):
        lifecycle.amend_minimum_success_rate(
            "steady_flow", campaign.source_name, batch.batch_id, minimum_success_rate=0.5, reason=" ", storage_root=storage
        )
    common.paths.resolve_generation_selected_pool_path(root).write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="sealed selected-success pool"):
        lifecycle.amend_minimum_success_rate(
            "steady_flow", campaign.source_name, batch.batch_id, minimum_success_rate=0.5, reason="After sealing", storage_root=storage
        )


def test_only_a_plan_campaign_without_an_explicit_cap_records_a_derived_admission_cap(generation_config_factory: Any) -> None:
    """A Dataset plan's campaign records null, which follows its rates; an explicit cap and a campaign without a plan keep their integer."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2, max_admission_cases=4)

    def recorded(**kwargs: Any) -> Any:
        return generation.cases.config.load_campaign_config(config_path, **kwargs).execution_values["submission"]["max_admission_cases"]

    assert recorded(authored_value=_authored(config_path, 0.8)) is None
    assert recorded(authored_value=_authored(config_path, 0.8), execution_overrides={"max_admission_cases": 4}) == 4
    assert recorded() == 4


@pytest.mark.parametrize("index", [True, 1.0])
def test_an_amendment_index_must_be_a_non_boolean_integer(generation_config_factory: Any, tmp_path: Path, index: object) -> None:
    """A JSON ``true`` or ``1.0`` amendment index is malformed history, never amendment one."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, 0.8))
    storage = tmp_path / "scientific-storage"
    (path,) = lifecycle.create_campaign(campaign, storage_root=storage).rglob("amendment_*.json")
    common.serialization.atomic_write_json(path, {**json.loads(path.read_text(encoding="utf-8")), "amendment_index": index})
    with pytest.raises(ValueError, match="amendments are malformed"):
        lifecycle.replacement_policy("steady_flow", campaign.source_name, storage_root=storage)


def test_a_source_without_a_plan_starts_at_one_and_is_lowered_through_its_campaign_handle(
    generation_config_factory: Any,
    tmp_path: Path,
) -> None:
    """A plain Source records nothing and uses s = 1; amend-replacements lowers one original batch and a late plan rate is refused."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = generation.cases.config.load_campaign_config(config_path)
    storage = tmp_path / "scientific-storage"
    root = lifecycle.create_campaign(campaign, storage_root=storage)
    assert not list(root.rglob("amendment_*.json"))
    (entry,) = lifecycle.replacement_policy("steady_flow", campaign.source_name, storage_root=storage)
    assert (entry["minimum_success_rate"], entry["amendments"]) == (1.0, [])
    run_id = f"source~steady_flow~{campaign.source_name}"
    amended = generation.campaign.amend_minimum_success_rate(
        run_id, "lentil", "natural", minimum_success_rate=0.5, reason="Accept a known failure", storage_root=storage
    )
    assert (amended["previous_minimum_success_rate"], amended["minimum_success_rate"]) == (1.0, 0.5)
    lifecycle.create_campaign(campaign, storage_root=storage)
    late = generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, 0.9))
    with pytest.raises(ValueError, match=r"records None.*requests 0\.9"):
        lifecycle.create_campaign(late, storage_root=storage)
    with pytest.raises(ValueError, match="no unique original batch"):
        generation.campaign.amend_minimum_success_rate(
            run_id, "lentil", "parameter_ood", minimum_success_rate=0.5, reason="No such batch", storage_root=storage
        )
    with pytest.raises(ValueError, match="canonical source campaign"):
        generation.campaign.amend_minimum_success_rate(
            "legacy-run", "lentil", "natural", minimum_success_rate=0.5, reason="No handle", storage_root=storage
        )


@pytest.mark.parametrize("rate", [0, 1.5, True, {"lentil": 0.9}])
def test_a_plan_rate_is_validated_and_belongs_to_plans(generation_config_factory: Any, rate: Any) -> None:
    """Reject a rate outside (0, 1], a boolean or a mapping, and a rate written into a campaign file."""
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    with pytest.raises(generation.cases.config.GenerationConfigError, match="0 < s <= 1"):
        generation.cases.config.load_campaign_config(config_path, authored_value=_authored(config_path, rate))
    config_path.write_text(yaml.safe_dump(_authored(config_path, 0.9), sort_keys=False), encoding="utf-8")
    with pytest.raises(generation.cases.config.GenerationConfigError, match="pre-registered by a Dataset plan"):
        generation.cases.config.load_campaign_config(config_path)
