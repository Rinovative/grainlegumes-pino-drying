"""Source names bind original design and physical domain, independent of selection.

Responsibilities:
  - Check canonical source names, material order, smoke namespaces, and semantic conflicts

Design principles:
  - Original design and physical domain determine identity independently of selection

This module does NOT:
  - Generate source cases or assign experiment roles
"""

from typing import Any

import pytest
import yaml

from src import common
from src.generation import generation_campaign_lifecycle as lifecycle
from src.generation.cases import generation_cases_config as config_service
from src.generation.cases import generation_cases_sampling as sampling_service
from src.generation.cases import generation_cases_seeding as seeding
from src.generation.contracts import generation_contracts_campaign_identity as identity
from src.generation.publication import generation_publication_campaign_evidence as evidence


def test_source_name_uses_registered_material_order_and_original_design() -> None:
    name = identity.canonical_campaign_name(
        {"chickpea": 50, "lentil": 50, "kidney_bean": 20},
        domain_profile="reference_range",
        mode="lhs",
        root_seed=9200,
        horizon_hours=None,
    )
    assert name == "lentil-50+chickpea-50+kidney-bean-20__ref-range__lhs-seed-9200"
    extended = identity.canonical_campaign_name(
        {"lentil": 3}, domain_profile="reference_and_parameter_extension", mode="sobol", root_seed=7, horizon_hours=None
    )
    assert extended == "lentil-3__ref-param-ext__sobol-seed-7"
    assert identity.horizon_token(4.5) == "h-4p5"
    with pytest.raises(ValueError, match="positive finite"):
        identity.horizon_token(0)


def test_only_a_horizon_deviating_from_the_configured_one_enters_the_name() -> None:
    """The configured horizon is part of the simulation definition; a short test run is named by its own horizon."""
    assert identity.deviating_horizon_hours(168, 168.0) is None
    deviating = identity.deviating_horizon_hours(4.0, 168.0)
    assert deviating == 4.0
    name = identity.canonical_campaign_name({"lentil": 3}, domain_profile="reference_range", mode="lhs", root_seed=7, horizon_hours=deviating)
    assert name == "lentil-3__ref-range__lhs-seed-7__h-4"


def test_replacement_batch_locators_extend_the_original_name_and_regime_by_the_candidate_ordinal() -> None:
    """One resolver owns the batch name and locator regime of originals and canonical replacements."""
    assert config_service.resolve_batch_locators("steady_flow", "lentil", "natural", {}) == ("steady_flow__lentil__natural", "natural")
    design = {"schema_version": sampling_service.SUPPLEMENTAL_DESIGN_SCHEMA_VERSION, "candidate_ordinal": 3}
    assert config_service.resolve_batch_locators("steady_flow", "lentil", "natural", {"supplemental_design": design}) == (
        "steady_flow__lentil__natural__replacement_0003",
        "natural_replacement_0003",
    )


@pytest.mark.parametrize("ordinal", [0, -1, True, "1", 1.0, None])
def test_replacement_batch_locators_reject_an_invalid_candidate_ordinal(ordinal: Any) -> None:
    """A canonical replacement without an integer ordinal of at least one has no batch identity."""
    design = {"schema_version": sampling_service.SUPPLEMENTAL_DESIGN_SCHEMA_VERSION, "candidate_ordinal": ordinal}
    with pytest.raises(ValueError, match="invalid candidate ordinal"):
        config_service.resolve_batch_locators("steady_flow", "lentil", "natural", {"supplemental_design": design})


def test_domain_token_is_the_short_path_safe_name_of_every_supported_profile() -> None:
    assert identity.domain_token("reference_range") == "ref-range"
    assert identity.domain_token("reference_and_parameter_extension") == "ref-param-ext"
    tokens = {profile: identity.domain_token(profile) for profile in identity.SUPPORTED_DOMAIN_PROFILES}
    assert len(set(tokens.values())) == len(tokens)
    with pytest.raises(ValueError, match="Unsupported source domain profile"):
        identity.domain_token("unlisted_profile")


def test_source_name_rejects_unsupported_material_domain_and_sampler() -> None:
    with pytest.raises(ValueError, match="registered"):
        identity.canonical_campaign_name({"unmeasured": 2}, domain_profile="reference_range", mode="lhs", root_seed=1, horizon_hours=None)
    with pytest.raises(ValueError, match="domain profile"):
        identity.canonical_campaign_name({"lentil": 2}, domain_profile="id", mode="lhs", root_seed=1, horizon_hours=None)
    with pytest.raises(ValueError, match="Unsupported campaign sampling mode"):
        identity.canonical_campaign_name({"lentil": 2}, domain_profile="reference_range", mode="compose", root_seed=1, horizon_hours=None)


def test_same_name_different_semantics_conflicts() -> None:
    with pytest.raises(ValueError, match="conflicting frozen scientific semantics"):
        identity.assert_campaign_semantics("lentil-2__ref-range__lhs-seed-1", "a" * 64, "b" * 64)


def test_authored_source_resolves_role_neutral_design(generation_config_factory: Any) -> None:
    config_path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    campaign = config_service.load_campaign_config(config_path, require_executable=False)
    assert campaign.campaign_name == config_path.stem
    assert campaign.domain_profile == "reference_range"
    assert campaign.selected_targets == {"lentil": 2}
    assert campaign.total_case_count == 2
    assert all(batch.scientific_values["domain_profile"] == "reference_range" for batch in campaign.batches)
    assert "material_role" not in campaign.batches[0].scientific_values
    assert "evaluation_regime" not in campaign.batches[0].scientific_values


def test_ordinary_run_overrides_resolve_small_design_and_preserve_schedule(generation_config_factory: Any) -> None:
    """Finite overrides change resolved science without changing authored configuration."""
    config_path, _ = generation_config_factory(simulation_profile="transient_drying", natural_count=6)
    authored = config_path.read_bytes()
    baseline = config_service.load_campaign_config(config_path, require_executable=False)
    overrides = {"materials": ["lentil"], "cases_per_material": 4, "duration_hours": 4.0}
    campaign = config_service.load_campaign_config(config_path, require_executable=False, scientific_overrides=overrides)
    scientific = campaign.batches[0].scientific_values
    assert config_path.read_bytes() == authored
    assert campaign.total_case_count == 4
    assert campaign.selected_targets == {"lentil": 4}
    assert campaign.root_seed == baseline.root_seed
    assert scientific["grid"] == baseline.batches[0].scientific_values["grid"]
    assert scientific["time"]["stop"] == 4.0
    assert scientific["time"]["regular_times"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert scientific["schedule_time"] == baseline.batches[0].scientific_values["time"]
    assert campaign.scientific_overrides == overrides
    assert campaign.campaign_digest != baseline.campaign_digest


def test_physical_horizons_name_their_own_sources_and_share_seeds(generation_config_factory: Any, tmp_path: Any) -> None:
    """A deviating horizon is another named Source without a revision, and keeps the seeds of the same design."""
    path, _template = generation_config_factory(simulation_profile="transient_drying", natural_count=4)
    full = config_service.load_campaign_config(path, require_executable=False)
    short = config_service.load_campaign_config(path, require_executable=False, scientific_overrides={"duration_hours": 4.0})
    assert "__h-" not in full.campaign_name
    assert short.campaign_name == full.campaign_name + "__h-4"
    assert full.root_seed == short.root_seed
    assert full.batches[0].seed_base == short.batches[0].seed_base
    assert full.batches[0].case_seed(1) == short.batches[0].case_seed(1)
    assert sampling_service.sample_case(full.batches[0], 1).values == sampling_service.sample_case(short.batches[0], 1).values
    roots = [lifecycle.create_campaign(campaign, storage_root=tmp_path / "storage") for campaign in (full, short)]
    assert roots[0] != roots[1]
    keys = {lifecycle.load_campaign_plan(c.profile.id, c.source_name, storage_root=tmp_path / "storage")["seed_key"] for c in (full, short)}
    assert len(keys) == 1
    expected_seed = seeding.derive_seed(full.root_seed, "campaign_candidate", full.profile.id, keys.pop(), "lentil", "primary", "original:1")
    for campaign in (full, short):
        reservation = lifecycle.reserve_candidate(
            campaign.profile.id,
            campaign.source_name,
            "lentil",
            origin="primary",
            design_key="original:1",
            candidate_position=1,
            target_slot=1,
            sample=sampling_service.sample_case(campaign.batches[0], 1).values,
            derived_seeds={"case_seed": campaign.batches[0].case_seed(1)},
            storage_root=tmp_path / "storage",
        )
        assert reservation["candidate_seed"] == expected_seed


def test_candidate_seeds_do_not_depend_on_the_campaign_name_builder(
    generation_config_factory: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Renaming campaigns leaves candidate seeds and the frozen seed key untouched."""
    path, _template = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    original = identity.canonical_campaign_name
    observed = []
    for index, builder in enumerate((original, lambda *args, **kwargs: f"renamed-{original(*args, **kwargs)}")):
        monkeypatch.setattr(identity, "canonical_campaign_name", builder)
        campaign = config_service.load_campaign_config(path, require_executable=False)
        storage = tmp_path / f"storage-{index}"
        lifecycle.create_campaign(campaign, storage_root=storage)
        reservation = lifecycle.reserve_candidate(
            campaign.profile.id,
            campaign.source_name,
            "lentil",
            origin="primary",
            design_key="original:1",
            candidate_position=1,
            target_slot=1,
            sample=sampling_service.sample_case(campaign.batches[0], 1).values,
            derived_seeds={"case_seed": campaign.batches[0].case_seed(1)},
            storage_root=storage,
        )
        plan = lifecycle.load_campaign_plan(campaign.profile.id, campaign.source_name, storage_root=storage)
        observed.append((campaign.source_name, plan["seed_key"], reservation["candidate_seed"]))
    assert observed[0][0] != observed[1][0]
    assert observed[0][1:] == observed[1][1:]


def test_normal_resource_overrides_preserve_science_and_persist_effective_execution(
    generation_config_factory: Any,
    tmp_path: Any,
) -> None:
    """Resolve a bounded allocation through normal layers without changing sampling."""
    path, _ = generation_config_factory(simulation_profile="transient_drying", natural_count=4, timeout=600, node_job_wall_time="01:00:00")
    authored_bytes = path.read_bytes()
    original = config_service.load_campaign_config(path)
    overrides = {"node_jobs": 1, "max_admission_cases": 4, "cores_per_case": 8, "case_timeout_seconds": 1500, "no_retry": True}
    bounded = config_service.load_campaign_config(path, execution_overrides=overrides)
    assert bounded.campaign_digest == original.campaign_digest
    assert bounded.source_name == original.source_name
    assert bounded.batches[0].seed_base == original.batches[0].seed_base
    assert bounded.batches[0].assignments == original.batches[0].assignments
    assert path.read_bytes() == authored_bytes
    values = bounded.execution_values
    # The case timeout never derives a node-job time: the authored 01:00:00 stays and merely has to fit 1500 s plus the margin.
    assert values["cluster"]["node_job_wall_time"] == "01:00:00"
    assert values["cluster"]["cores_per_case"] == 8
    assert values["cluster"]["node_jobs"] == 1
    assert values["cluster"]["concurrent_cases_per_node"] == 3
    assert values["runtime"]["timeout_seconds"] == 1500
    assert values["runtime"]["graceful_stop_reserve_seconds"] == 300
    assert values["runtime"]["license_wait"]["enabled"] is False
    assert values["runtime"]["license_wait"]["in_allocation_retry"]["enabled"] is False
    assert values["submission"]["max_admission_cases"] == 4
    lifecycle.create_campaign(bounded, storage_root=tmp_path)
    plan = lifecycle.load_campaign_plan(bounded.profile.id, bounded.source_name, storage_root=tmp_path)
    assert plan["execution_overrides"] == overrides
    assert plan["execution_values"] == values
    assert plan["batches"][0]["execution_values"] == values


@pytest.mark.parametrize(
    "overrides",
    [
        {"case_timeout_seconds": 0},
        {"case_timeout_seconds": float("inf")},
        {"node_jobs": 0},
        {"node_jobs": 5},
        {"case_timeout_seconds": 5000},
        {"cores_per_case": True},
        {"no_retry": "yes"},
    ],
)
def test_normal_resource_overrides_reject_invalid_values(generation_config_factory: Any, overrides: Any) -> None:
    """Reject malformed budgets before creating inputs or submitting jobs."""
    path, _ = generation_config_factory()
    with pytest.raises(config_service.GenerationConfigError):
        config_service.load_campaign_config(path, execution_overrides=overrides)


def test_source_name_is_the_label_and_validates_the_digest_it_is_frozen_with() -> None:
    """The label names the folder as written; the digest is validated, never embedded."""
    label = "lentil-2__ref-range__lhs-seed-17"
    assert identity.source_name(label, "a" * 64) == label
    assert identity.source_name(label, "b" * 64) == label
    with pytest.raises(ValueError, match="SHA-256"):
        identity.source_name(label, "invalid")
    with pytest.raises(ValueError, match="filesystem component limit"):
        identity.source_name("a" * 256, "a" * 64)


@pytest.mark.parametrize(
    ("label", "valid"),
    [
        ("fix", True),
        ("converter-fix", True),
        ("v2_a+b", True),
        ("a1", True),
        ("Upper", False),
        ("dot.v2", False),
        ("-lead", False),
        ("trail-", False),
        ("a--b", False),
        ("a__b", False),
        ("nested/label", False),
        ("", False),
    ],
)
def test_revision_grammar_is_one_rule_and_the_revised_label_is_a_valid_source_name(label: str, valid: bool) -> None:
    """A revision is lowercase letters and digits joined by single separators; only such labels form a Source name."""
    design = "lentil-2__ref-range__lhs-seed-17"
    if not valid:
        with pytest.raises(ValueError, match="Revision"):
            identity.revision_label(label)
        with pytest.raises(ValueError, match="Revision"):
            identity.revised_source_label(design, label)
        return
    assert identity.revision_label(label) == label
    revised = identity.revised_source_label(design, label)
    assert revised == f"{design}__rev-{label}"
    assert identity.source_name(revised, "a" * 64) == revised


@pytest.mark.parametrize("value", [None, 7, ["fix"]])
def test_revision_must_be_text(value: object) -> None:
    with pytest.raises(ValueError, match="Revision"):
        identity.revision_label(value)


def test_campaign_revision_key_labels_the_source_and_the_old_free_label_is_refused(generation_config_factory: Any) -> None:
    """The authored ``revision`` names the Source ``<name>__rev-LABEL`` without touching seeds or the digest; ``source_name`` no longer exists."""
    path, _ = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    authored = yaml.safe_load(path.read_text())
    plain = config_service.load_campaign_config(path)
    revised = config_service.load_campaign_config(path, authored_value={**authored, "revision": "fix-1"})

    assert plain.source_name == plain.campaign_name
    assert revised.source_name == f"{plain.campaign_name}__rev-fix-1" == identity.revised_source_label(plain.campaign_name, "fix-1")
    assert (revised.campaign_name, revised.campaign_digest) == (plain.campaign_name, plain.campaign_digest)
    assert [(batch.batch_id, batch.seed_base, batch.assignments) for batch in revised.batches] == [
        (batch.batch_id, batch.seed_base, batch.assignments) for batch in plain.batches
    ]
    assert lifecycle._resolved_plan(revised)["seed_key"] == lifecycle._resolved_plan(plain)["seed_key"]
    with pytest.raises(config_service.GenerationConfigError, match=r"unknown=\['source_name'\]"):
        config_service.load_campaign_config(path, authored_value={**authored, "source_name": "free-label"})
    with pytest.raises(config_service.GenerationConfigError, match="Revision"):
        config_service.load_campaign_config(path, authored_value={**authored, "revision": "Free Label"})


def test_scientific_revision_override_equals_the_authored_key(generation_config_factory: Any) -> None:
    """The run override ``revision`` reaches the loader as the campaign value and is recorded with the other scientific overrides."""
    path, _ = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    authored = yaml.safe_load(path.read_text())
    by_override = config_service.load_campaign_config(path, scientific_overrides={"revision": "fix-1"})
    by_key = config_service.load_campaign_config(path, authored_value={**authored, "revision": "fix-1"})

    assert by_override.source_name == by_key.source_name
    assert by_override.campaign_digest == by_key.campaign_digest
    assert by_override.scientific_overrides == {"revision": "fix-1"}
    assert lifecycle._resolved_plan(by_override)["scientific_overrides"] == {"revision": "fix-1"}
    assert lifecycle._resolved_plan(by_override)["source_name"] == by_override.source_name
    with pytest.raises(config_service.GenerationConfigError, match="Revision"):
        config_service.load_campaign_config(path, scientific_overrides={"revision": "Bad Label"})


def test_seed_key_depends_on_scientific_design_and_not_on_the_display_name() -> None:
    """Candidate seeds survive renaming but change with counts, seed base, or purpose."""
    design = {
        "task": "transient_drying",
        "domain_profile": "reference_range",
        "sampling_method": "lhs",
        "seed_base": 9220,
        "purpose": "source_generation",
    }
    key = seeding.campaign_seed_key(design_sizes={"lentil": 3, "chickpea": 3}, **design)
    assert key == seeding.campaign_seed_key(design_sizes={"chickpea": 3, "lentil": 3}, **design)
    assert key != seeding.campaign_seed_key(design_sizes={"lentil": 4, "chickpea": 3}, **design)
    assert key != seeding.campaign_seed_key(design_sizes={"lentil": 3, "chickpea": 3}, **{**design, "seed_base": 9221})
    assert key != seeding.campaign_seed_key(design_sizes={"lentil": 3, "chickpea": 3}, **{**design, "purpose": "other"})


def test_dataset_plan_envelope_reuses_source_and_preserves_frozen_authorship(
    generation_config_factory: Any,
    tmp_path: Any,
) -> None:
    """Roles, splits and plan location remain outside the physical Source identity."""
    path, _ = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    authored = yaml.safe_load(path.read_text())
    for key in ("sources_config", "registry_config", "common_config", "operations_config", "profile_config", "execution_config"):
        authored[key] = str((path.parent / authored[key]).resolve())
    authored["revision"] = "my-readable-source"
    plans = common.paths.get_project_root() / "configs/02_datasets"
    plans.mkdir(parents=True)
    first = plans / "first.yaml"
    second = plans / "second.yaml"
    first.write_text("roles: {lentil: train}\nsplit_seed: 11\n")
    second.write_text("roles: {lentil: test}\nsplit_seed: 29\n")
    before = first.read_bytes(), second.read_bytes()
    campaigns = [config_service.load_campaign_config(p, authored_value=authored) for p in (first, second)]
    label = f"{campaigns[0].campaign_name}__rev-my-readable-source"
    assert campaigns[0].source_name == campaigns[1].source_name == label
    assert campaigns[0].campaign_digest == campaigns[1].campaign_digest
    assert campaigns[0].batches[0].case_seed(1) == campaigns[1].batches[0].case_seed(1)
    storage = tmp_path / "storage"
    root = lifecycle.create_campaign(campaigns[0], storage_root=storage)
    frozen = (root / "_record/campaign_plan.json").read_bytes()
    assert lifecycle.create_campaign(campaigns[1], storage_root=storage) == root
    assert (root / "_record/campaign_plan.json").read_bytes() == frozen
    plan = lifecycle.load_campaign_plan("steady_flow", label, storage_root=storage)
    assert plan["authored_value"] == authored
    original = next(record for record in plan["source_files"] if record["relative_path"] == "configs/02_datasets/first.yaml")
    assert original["content"] == before[0].decode()
    assert (first.read_bytes(), second.read_bytes()) == before
    manifest = {
        "campaign_run_id": f"source~steady_flow~{label}",
        "remote_storage_root": str(storage),
        "source_name": campaigns[0].source_name,
        "simulation_profile": "steady_flow",
        "campaign_config": "configs/02_datasets/first.yaml",
        "execution_config_digest": common.serialization.canonical_json_sha256(plan["execution_values"]),
        "selected_batch_names": [batch.batch_name for batch in campaigns[0].batches],
        **{key: plan[key] for key in ("campaign_name", "campaign_digest", "campaign_id")},
    }
    first.unlink()
    restored = lifecycle.campaign_from_frozen_run(manifest, storage_root=storage)
    assert restored.source_name == label
    assert restored.authored_value == authored
    assert restored.campaign_digest == campaigns[0].campaign_digest
    assert restored.batches[0].case_seed(1) == campaigns[0].batches[0].case_seed(1)


def test_authored_campaign_locator_remains_contained_yaml_without_symlinks(
    generation_config_factory: Any,
) -> None:
    """Dataset plans are admitted while escaping paths and links fail closed."""
    generation_config_factory(simulation_profile="steady_flow", natural_count=1)
    root = common.paths.get_project_root()
    plan = root / "configs/02_datasets/plan.yaml"
    plan.parent.mkdir(parents=True)
    plan.write_text("plan: evidence\n")
    assert evidence.resolve_campaign_config_path("configs/02_datasets/plan.yaml") == plan
    assert evidence.resolve_frozen_campaign_config_locator("configs/02_datasets/plan.yaml") == plan
    link = plan.with_name("linked.yaml")
    link.symlink_to(plan.name)
    for unsafe in ("configs/../plan.yaml", str(plan), "configs/02_datasets/plan.json", "configs/02_datasets/linked.yaml"):
        with pytest.raises(ValueError, match=r"canonical generation campaign path|symlink"):
            evidence.resolve_campaign_config_path(unsafe)
        with pytest.raises(ValueError, match=r"unsafe|escapes"):
            evidence.resolve_frozen_campaign_config_locator(unsafe)
