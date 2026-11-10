"""Protect the shared curated W&B history sections and the run organization of both tasks.

Responsibilities:
  - Check that steady and transient runs expose the design's sections, key order and conditional series, and that the
    alphabetical name order equals the position order of the pages of six
  - Check the Optuna keys and which series stay in the run summary
  - Check group, job type and name of steady and transient training runs

Design principles:
  - The design table is the contract: expected key sets are written out, never read back from the producer

This module does NOT:
  - Contact W&B or define model optimization behavior
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest
from support import configs

from src.experiments import experiments_wandb_schema as schema

_OBJECTIVE = "normalized_drying_group_macro_rmse"
_BOTH_PARTS = ("material_ood", "parameter_ood")
_FIELDS = ("phi", "temperature", "w_surf")
_TERMS = ("boundary", "continuity", "momentum")
_PERFORMANCE = [
    "duration-epoch",
    "duration-train",
    "duration-validation",
    "memory-cuda_peak_gib",
    "progress-{}",
    "rate-samples_per_second",
]
_OVERVIEW = [
    "objective-id",
    "objective-material",
    "objective-parameter",
    "train-loss_total",
    "transfer_gap-material",
    "transfer_gap-parameter",
]


def _pages(quantities: tuple[str, ...]) -> list[str]:
    """Return the names of a comparison section: ID above OOD on the material page, then on the parameter page."""
    return [f"{part}-{label}-{name}" for part in ("material", "parameter") for label in ("id", "ood") for name in quantities]


def _sections(projections: tuple[schema.MetricProjection, ...]) -> dict[str, list[str]]:
    """Group the destination names by their section in emission order; every key has exactly one slash."""
    sections: dict[str, list[str]] = defaultdict(list)
    for projection in projections:
        section, slash, name = projection.wandb_key.partition("/")
        assert slash == "/"
        assert "/" not in name
        sections[section].append(name)
    return dict(sections)


def _steady(**overrides: Any) -> tuple[schema.MetricProjection, ...]:
    """Return the steady projections of a physics-informed GPU run with both OOD parts unless overridden."""
    metrics = configs.resolve(configs.direct_config())["evaluation"]["metrics"]
    flags: dict[str, Any] = {
        "physics_training_enabled": True,
        "physics_monitor_enabled": True,
        "state_aux_enabled": False,
        "ood_roles": _BOTH_PARTS,
        "cuda_enabled": True,
        "optuna_trial": False,
    } | overrides
    return schema.curated_metric_projections("steady_flow", metrics, objective_id="normalized_group_macro_rmse", **flags)


def _transient(**overrides: Any) -> tuple[schema.MetricProjection, ...]:
    """Return the transient projections of a GPU run with state aux and both OOD parts unless overridden."""
    flags: dict[str, Any] = {
        "physics_training_enabled": False,
        "physics_monitor_enabled": False,
        "state_aux_enabled": True,
        "ood_roles": _BOTH_PARTS,
        "cuda_enabled": True,
        "optuna_trial": False,
    } | overrides
    metrics = [{"id": f"{space}_rmse_{field}"} for space in ("normalized", "physical") for field in ("T", "phi", "w_surf", "w_int")]
    return schema.curated_metric_projections("transient_drying", metrics, objective_id=_OBJECTIVE, **flags)


def test_steady_history_follows_the_design_sections() -> None:
    """Fill every section for a physics-informed run, in the position order of its pages of six."""
    sections = _sections(_steady())

    assert sections == {
        "Overview": _OVERVIEW,
        "Training": ["learning_rate", "loss_data", "optimizer_steps", *(f"weight-{term}" for term in _TERMS)],
        "Physics": [f"{role}-{term}" for role in ("train", "val") for term in _TERMS],
        "Fields": _pages(("p", "u", "v")),
        "Physical": _pages(("p", "u", "v")),
        "Norms": _pages(("relative_h1", "relative_l2", "velocity")),
        "Performance": [name.format("remaining_seconds") for name in _PERFORMANCE],
    }


def test_transient_history_follows_the_design_sections() -> None:
    """Fill every section of a transient run, including the one-step page of each role above its rollout series."""
    sections = _sections(_transient())

    assert sections == {
        "Overview": _OVERVIEW,
        "Training": [
            "curriculum-horizon",
            "curriculum-progress",
            "forward_transitions",
            "learning_rate",
            "optimizer_steps",
            "transitions",
        ],
        "Loss": ["data-phi", "data-temperature", "data-total", "data-w_int", "data-w_surf", "state_aux"],
        "Fields": _pages(_FIELDS),
        "Physical": _pages(_FIELDS),
        "Moisture": _pages(("grain_moisture", "w_gr_physical", "w_int")),
        "OneStep": [f"{role}-{kind}-{name}" for role in ("id", "material", "parameter") for kind in ("one_step", "rollout") for name in _FIELDS],
        "Performance": [name.format("wall_seconds") for name in _PERFORMANCE],
    }


@pytest.mark.parametrize("projections", [_steady(optuna_trial=True), _transient(optuna_trial=True)], ids=["steady", "transient"])
def test_names_sort_into_their_position_order(projections: tuple[schema.MetricProjection, ...]) -> None:
    """The order of a section equals its emission order under byte-wise, case-insensitive and punctuation-ignoring collation."""
    for section, names in _sections(projections).items():
        assert names == sorted(names), section
        assert names == sorted(names, key=str.casefold), section
        assert names == sorted(names, key=lambda name: name.replace("-", "").replace("_", "")), section
        if section in {"Fields", "Physical", "Norms", "Moisture", "OneStep"}:
            assert len(names) % 6 == 0, section


def test_series_without_a_producer_are_not_listed() -> None:
    """Drop OOD, physics, state aux and CUDA series when the run does not produce them."""
    steady = _sections(_steady(physics_training_enabled=False, physics_monitor_enabled=False, ood_roles=(), cuda_enabled=False))
    transient = _sections(_transient(state_aux_enabled=False, ood_roles=(), cuda_enabled=False))

    assert set(steady) == {"Overview", "Training", "Fields", "Physical", "Norms", "Performance"}
    assert steady["Overview"] == ["objective-id", "train-loss_total"]
    assert steady["Training"] == ["learning_rate", "optimizer_steps"]
    assert transient["Loss"] == ["data-phi", "data-temperature", "data-total", "data-w_int", "data-w_surf"]
    assert transient["OneStep"] == [f"id-{kind}-{name}" for kind in ("one_step", "rollout") for name in _FIELDS]
    assert "memory-cuda_peak_gib" not in steady["Performance"]


@pytest.mark.parametrize(("ood_roles", "part"), [(("material_ood",), "material"), (("parameter_ood",), "parameter")])
def test_a_missing_ood_part_drops_its_page(ood_roles: tuple[str, ...], part: str) -> None:
    """A configured part adds its Overview series and its page; the other part adds nothing, so no page 2 shows OOD without ID."""
    other = "parameter" if part == "material" else "material"
    sections = _sections(_steady(ood_roles=ood_roles))

    assert sections["Fields"] == [f"{part}-{label}-{name}" for label in ("id", "ood") for name in "puv"]
    assert sections["Overview"] == ["objective-id", f"objective-{part}", "train-loss_total", f"transfer_gap-{part}"]
    assert not any(name.startswith(f"{other}-") or name.endswith(f"-{other}") for names in sections.values() for name in names)
    with pytest.raises(ValueError, match="unknown OOD roles"):
        _steady(ood_roles=("additional",))


def test_without_an_ood_part_the_comparison_sections_show_the_id_series_alone() -> None:
    """Keep the ID errors per field visible as ``id-<q>`` when no OOD part is configured."""
    steady = _sections(_steady(ood_roles=()))
    transient = _sections(_transient(ood_roles=()))

    assert steady["Fields"] == ["id-p", "id-u", "id-v"]
    assert steady["Norms"] == ["id-relative_h1", "id-relative_l2", "id-velocity"]
    assert transient["Fields"] == ["id-phi", "id-temperature", "id-w_surf"]
    assert transient["Moisture"] == ["id-grain_moisture", "id-w_gr_physical", "id-w_int"]
    assert transient["OneStep"] == [f"id-{kind}-{name}" for kind in ("one_step", "rollout") for name in _FIELDS]
    for names in (*steady.values(), *transient.values()):
        assert names == sorted(names)


def test_transient_metric_series_follow_the_resolved_metrics() -> None:
    """List a transient Physical or OneStep series only when the resolved metrics contain its metric id."""
    projections = schema.curated_metric_projections(
        "transient_drying",
        [{"id": "normalized_rmse_phi"}, {"id": "physical_rmse_phi"}],
        objective_id=_OBJECTIVE,
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        state_aux_enabled=False,
        ood_roles=(),
        cuda_enabled=False,
        optuna_trial=False,
    )
    sections = _sections(projections)

    assert sections["Physical"] == ["id-phi"]
    assert sections["OneStep"] == ["id-one_step-phi", "id-rollout-phi"]
    assert sections["Fields"] == ["id-phi", "id-temperature", "id-w_surf"]  # objective components need no metric id


def test_steady_evaluation_series_follow_the_resolved_metrics() -> None:
    """List a diagnostic only when the resolved evaluation metrics contain it."""
    metrics = [metric for metric in configs.resolve(configs.direct_config())["evaluation"]["metrics"] if metric["id"] != "normalized_relative_l2"]
    projections = schema.curated_metric_projections(
        "steady_flow",
        metrics,
        objective_id="normalized_group_macro_rmse",
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        state_aux_enabled=False,
        ood_roles=("material_ood",),
        cuda_enabled=False,
        optuna_trial=False,
    )

    assert _sections(projections)["Norms"] == ["material-id-relative_h1", "material-id-velocity", "material-ood-relative_h1", "material-ood-velocity"]


@pytest.mark.parametrize("task", ["steady", "transient"])
def test_destinations_are_unique_and_id_feeds_both_pages(task: str) -> None:
    """Map each destination once while an ID source feeds both pages; only the Optuna series keeps its last value in the summary."""
    projections = (_steady if task == "steady" else _transient)(optuna_trial=True)
    destinations = [projection.wandb_key for projection in projections]
    by_destination = {projection.wandb_key: projection for projection in projections}
    first = "Fields/material-id-p" if task == "steady" else "Fields/material-id-temperature"

    assert len(destinations) == len(set(destinations))
    assert by_destination[first].source_key == by_destination[first.replace("material", "parameter")].source_key
    assert _sections(projections)["Optuna"] == ["best_objective"]
    assert {key for key, projection in by_destination.items() if projection.in_summary} == {"Optuna/best_objective"}
    assert by_destination["Performance/memory-cuda_peak_gib"].multiplier * 1024**3 == 1.0
    assert "Optuna" not in _sections(_steady())
    assert "Optuna" not in _sections(_transient())


def test_producer_source_keys_feed_the_curated_series() -> None:
    """Read the loops' own source names so a series never stays defined but empty."""
    steady = {projection.source_key for projection in _steady()}
    transient = {projection.source_key for projection in _transient()}
    assert {
        "id/normalized_rmse_p",
        "material_ood/normalized_rmse_p",
        "parameter_ood/normalized_rmse_p",
        "physics/id/flux_balance_mse",
        "physics/train/continuity_weight",
        "global_step",
        "system/estimated_remaining_seconds",
    } <= steady
    assert {
        f"id/{_OBJECTIVE}/component/grain_moisture_error",
        "material_ood/physical/w_gr_rmse",  # the loop's physical grain-moisture RMSE
        "parameter_ood/physical_rmse_T",
        "id/guardrail/one_step/normalized_rmse_phi",  # the loop's one-step guardrail of the ID role
        "material_ood/normalized_rmse_w_surf",
        "transient/wall_seconds",  # the adapter telemetry name of the cumulative stage wall time
        "transient/processed_target_transitions",  # the cumulative stage total, not the per-epoch transient/train/ count
        "transient/forward_transitions",  # the cumulative stage total of model forward transitions
        "transient/curriculum_progress",
        "system/id_evaluation_duration_seconds",
    } <= transient
    assert not any(source.startswith("ood/") for source in steady)


def test_summary_keeps_the_selected_keys_of_projected_series_only() -> None:
    """Derive the checkpoint-evaluation summary keys from the projection: evaluation, monitor and weight series, nothing else."""
    keys = {schema.selected_summary_key(projection.source_key) for projection in _steady()} - {None}

    assert {
        "selected/id/normalized_rmse_p",
        "selected/material_ood/objective_difference",
        "selected/parameter_ood/objective_difference",
        "selected/physics/flux_balance_mse",
        "selected/training/continuity_weight",
    } <= keys
    assert "selected/id/physical_velocity_vector_rmse" not in keys
    assert schema.selected_summary_key("train/loss_total") is None
    assert schema.selected_summary_key("physics/train/loss_momentum") is None


def test_projection_requires_an_objective() -> None:
    """Reject a projection without the objective that names its Overview series."""
    with pytest.raises(ValueError, match="objective"):
        schema.curated_metric_projections(
            "transient_drying",
            [],
            objective_id="",
            physics_training_enabled=False,
            physics_monitor_enabled=False,
            state_aux_enabled=False,
            ood_roles=_BOTH_PARTS,
            cuda_enabled=False,
            optuna_trial=False,
        )


def test_steady_training_run_is_grouped_by_family_and_config() -> None:
    """Group every seed of one config together and keep the seed in the run name only."""
    first = configs.resolve(configs.direct_config())
    other_seed = configs.direct_config()
    other_seed["run"]["seed"] = 18
    second = configs.resolve(other_seed)
    layout = first["run"]["layout"]

    group, job_type, name = schema.run_organization(first, workflow="train")
    other_group, _, other_name = schema.run_organization(second, workflow="train")

    assert group is not None
    assert group == other_group == f"{layout['family']}/{layout['config_name']}"
    assert (job_type, name) == ("train", first["run"]["name"])
    assert other_name != name
    assert "seed" not in group


def test_transient_stages_share_group_and_use_distinct_job_types() -> None:
    """Keep A0, B and A1 of one config in one group; opaque Dataset IDs stay out of it."""
    raw = configs.transient_matched_config()
    raw["data"]["train_dataset"] = "transient_drying__synthetic_family__id__0123456789abcdef"
    plan = configs.resolve_plan(raw)
    stages = {"a0": plan.a0, "b": plan.b, "a1": plan.a1}
    organization = {name: schema.run_organization(stage, workflow="train") for name, stage in stages.items()}
    layout = plan.a0["run"]["layout"]
    group = organization["a0"][0]

    assert group is not None
    assert group == f"{layout['family']}/{layout['config_name']}"
    assert {group_label for group_label, _job, _name in organization.values()} == {group}
    assert {name: job for name, (_group, job, _name) in organization.items()} == {"a0": "stage_a0", "b": "stage_b", "a1": "stage_a1"}
    assert {name: run_name for name, (_group, _job, run_name) in organization.items()} == {
        name: stage["run"]["name"] for name, stage in stages.items()
    }
    assert "0123456789abcdef" not in group
    assert "seed" not in group


def test_unsupported_transient_stage_is_rejected() -> None:
    """Fail closed on a stage and arm pair without a job type."""
    a0 = configs.resolve_plan(configs.transient_matched_config()).a0
    config = {**a0, "training": {**a0["training"], "stage": "b"}}  # stage b with the A0 arm is no transient stage

    with pytest.raises(ValueError, match="stage/arm"):
        schema.run_organization(config, workflow="train")


def test_other_workflows_have_no_group() -> None:
    """Leave acceptance workflows ungrouped under their own job type."""
    config = configs.resolve(configs.direct_config())

    assert schema.run_organization(config, workflow="tracking_validation") == (None, "tracking_validation", config["run"]["name"])
