"""Protect the shared curated W&B history sections and the run organization of both tasks.

Responsibilities:
  - Check that steady and transient runs expose the design's sections, key order and conditional series, and that the
    alphabetical name order equals the position order of the one-row pages
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
_FIELDS = ("phi", "temperature", "w_surf")
_TERMS = ("boundary", "continuity", "momentum")
_BUDGET = ["budget-progress"]
_PERFORMANCE = [
    "duration-event",
    "duration-train",
    "duration-validation",
    "memory-cuda_peak_gib",
    "progress-{}",
    "rate-units_per_second",
]
_OVERVIEW = ["objective-id_validation", "train-loss_total"]


def _pages(quantities: tuple[str, ...]) -> list[str]:
    """Return the names of a comparison section: the ID validation series, one column per quantity."""
    return [f"id_validation-{name}" for name in quantities]


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
    """Return the steady projections of a physics-informed GPU run unless overridden."""
    metrics = configs.resolve(configs.direct_config())["evaluation"]["metrics"]
    flags: dict[str, Any] = {
        "physics_training_enabled": True,
        "physics_monitor_enabled": True,
        "cuda_enabled": True,
        "optuna_trial": False,
        "rollout_curriculum": False,
    } | overrides
    return schema.curated_metric_projections("steady_flow", metrics, objective_id="normalized_group_macro_rmse", **flags)


def _transient(**overrides: Any) -> tuple[schema.MetricProjection, ...]:
    """Return the transient projections of a self-fed (Stage B) GPU run unless overridden."""
    flags: dict[str, Any] = {
        "physics_training_enabled": False,
        "physics_monitor_enabled": False,
        "cuda_enabled": True,
        "optuna_trial": False,
        "rollout_curriculum": True,
    } | overrides
    metrics = [{"id": f"{space}_rmse_{field}"} for space in ("normalized", "physical") for field in ("T", "phi", "w_surf", "w_int")]
    metrics += [{"id": "physical_relative_h1"}, {"id": "physical_relative_l2"}]
    return schema.curated_metric_projections("transient_drying", metrics, objective_id=_OBJECTIVE, **flags)


def test_steady_history_follows_the_design_sections() -> None:
    """Fill every section for a physics-informed run, in the position order of its pages."""
    sections = _sections(_steady())

    assert sections == {
        "Overview": _OVERVIEW,
        "Training": [*_BUDGET, "learning_rate", "optimisation_seconds", "optimizer_steps", *(f"weight-{term}" for term in _TERMS)],
        "Loss": ["data-p", "data-total", "data-u", "data-v", "data-velocity", *(f"physics-{term}" for term in _TERMS)],
        "Residuals": [f"val-{term}" for term in _TERMS],
        "Fields": _pages(("p", "u", "v", "velocity")),
        "Physical": _pages(("p", "u", "v")),
        "Norms": _pages(("relative_h1", "relative_l2")),
        "Performance": [name.format("remaining_seconds") for name in _PERFORMANCE],
    }


def test_transient_history_follows_the_design_sections() -> None:
    """Fill every section of a transient run, including the one-step page above its rollout series."""
    sections = _sections(_transient())

    assert sections == {
        "Overview": _OVERVIEW,
        "Training": [
            *_BUDGET,
            "curriculum-horizon",
            "curriculum-progress",
            "forward_transitions",
            "learning_rate",
            "optimisation_seconds",
            "optimizer_steps",
            "transitions",
        ],
        "Loss": ["data-phi", "data-temperature", "data-total", "data-w_int", "data-w_surf"],
        "Fields": _pages(_FIELDS),
        "Physical": _pages(_FIELDS),
        "Norms": _pages(("relative_h1", "relative_l2")),
        "Moisture": _pages(("grain_moisture", "w_gr_physical", "w_int")),
        "OneStep": [f"id_validation-{kind}-{name}" for kind in ("one_step", "rollout") for name in _FIELDS],
        "Performance": [name.format("remaining_seconds") for name in _PERFORMANCE],
    }


@pytest.mark.parametrize("projections", [_steady(optuna_trial=True), _transient(optuna_trial=True)], ids=["steady", "transient"])
def test_names_sort_into_their_position_order(projections: tuple[schema.MetricProjection, ...]) -> None:
    """The order of a section equals its emission order under byte-wise, case-insensitive and punctuation-ignoring collation."""
    for section, names in _sections(projections).items():
        assert names == sorted(names), section
        assert names == sorted(names, key=str.casefold), section
        assert names == sorted(names, key=lambda name: name.replace("-", "").replace("_", "")), section
        if section in {"Fields", "Physical", "Norms", "Moisture", "OneStep"}:
            # One column per quantity: Norms has two quantities, steady Fields four, the others three (OneStep two rows of three).
            columns = 2 if section == "Norms" else 4 if section == "Fields" and any(name.endswith("-velocity") for name in names) else 3
            assert len(names) % columns == 0, section


def test_series_without_a_producer_are_not_listed() -> None:
    """Drop physics, CUDA and curriculum series when the run does not produce them."""
    steady = _sections(_steady(physics_training_enabled=False, physics_monitor_enabled=False, cuda_enabled=False))
    transient = _sections(_transient(cuda_enabled=False))
    teacher_forced = _sections(_transient(rollout_curriculum=False))

    assert set(steady) == {"Overview", "Training", "Loss", "Fields", "Physical", "Norms", "Performance"}
    assert steady["Overview"] == ["objective-id_validation", "train-loss_total"]
    assert steady["Training"] == [*_BUDGET, "learning_rate", "optimisation_seconds", "optimizer_steps"]
    # The data loss is produced by every run, so both projects show it under the same name.
    assert steady["Loss"] == ["data-p", "data-total", "data-u", "data-v", "data-velocity"]
    assert "data-total" in transient["Loss"]
    assert teacher_forced["Training"] == [*_BUDGET, "forward_transitions", "learning_rate", "optimisation_seconds", "optimizer_steps", "transitions"]
    assert transient["Loss"] == ["data-phi", "data-temperature", "data-total", "data-w_int", "data-w_surf"]
    assert transient["OneStep"] == [f"id_validation-{kind}-{name}" for kind in ("one_step", "rollout") for name in _FIELDS]
    assert "Residuals" not in transient
    assert "memory-cuda_peak_gib" not in steady["Performance"]


def test_the_comparison_sections_show_the_id_validation_series_alone() -> None:
    """Keep the ID errors per field visible as ``id_validation-<q>``, never as ``id-<q>``."""
    steady = _sections(_steady())
    transient = _sections(_transient())

    assert steady["Fields"] == ["id_validation-p", "id_validation-u", "id_validation-v", "id_validation-velocity"]
    assert steady["Norms"] == transient["Norms"] == ["id_validation-relative_h1", "id_validation-relative_l2"]
    assert transient["Fields"] == ["id_validation-phi", "id_validation-temperature", "id_validation-w_surf"]
    assert transient["Moisture"] == ["id_validation-grain_moisture", "id_validation-w_gr_physical", "id_validation-w_int"]
    assert transient["OneStep"] == [f"id_validation-{kind}-{name}" for kind in ("one_step", "rollout") for name in _FIELDS]
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
        cuda_enabled=False,
        optuna_trial=False,
        rollout_curriculum=False,
    )
    sections = _sections(projections)

    assert sections["Physical"] == ["id_validation-phi"]
    assert "Norms" not in sections
    assert sections["OneStep"] == ["id_validation-one_step-phi", "id_validation-rollout-phi"]
    assert sections["Fields"] == ["id_validation-phi", "id_validation-temperature", "id_validation-w_surf"]  # objective components need no metric id


def test_steady_evaluation_series_follow_the_resolved_metrics() -> None:
    """List a diagnostic only when the resolved evaluation metrics contain it."""
    metrics = [metric for metric in configs.resolve(configs.direct_config())["evaluation"]["metrics"] if metric["id"] != "physical_relative_l2"]
    projections = schema.curated_metric_projections(
        "steady_flow",
        metrics,
        objective_id="normalized_group_macro_rmse",
        physics_training_enabled=False,
        physics_monitor_enabled=False,
        cuda_enabled=False,
        optuna_trial=False,
        rollout_curriculum=False,
    )

    assert _sections(projections)["Norms"] == ["id_validation-relative_h1"]


@pytest.mark.parametrize("task", ["steady", "transient"])
def test_destinations_are_unique_and_only_optuna_keeps_its_summary_value(task: str) -> None:
    """Map each destination once; only the Optuna series keeps its last value in the summary."""
    projections = (_steady if task == "steady" else _transient)(optuna_trial=True)
    destinations = [projection.wandb_key for projection in projections]
    by_destination = {projection.wandb_key: projection for projection in projections}

    assert len(destinations) == len(set(destinations))
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
        "physics/id/flux_balance_mse",
        "physics/train/continuity_weight",
        "global_step",
        "system/estimated_remaining_seconds",
    } <= steady
    assert {
        f"id/{_OBJECTIVE}/component/grain_moisture",
        "id/physical/w_gr_rmse",  # the loop's physical grain-moisture RMSE
        "id/physical_rmse_T",
        "id/guardrail/one_step/normalized_rmse_phi",  # the loop's one-step guardrail of the ID role
        "system/estimated_remaining_seconds",  # both tasks' progress series, from the budget fraction
        "transient/processed_target_transitions",  # the cumulative stage total, not the per-epoch transient/train/ count
        "transient/forward_transitions",  # the cumulative stage total of model forward transitions
        "transient/curriculum_progress",
        "system/id_evaluation_duration_seconds",
    } <= transient


def test_summary_keeps_the_selected_keys_of_projected_series_only() -> None:
    """Derive the checkpoint-evaluation summary keys from the projection: evaluation, monitor and weight series, nothing else."""
    keys = {schema.selected_summary_key(projection.source_key) for projection in _steady()} - {None}

    assert {
        "selected/id/normalized_rmse_p",
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
            cuda_enabled=False,
            optuna_trial=False,
            rollout_curriculum=False,
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
