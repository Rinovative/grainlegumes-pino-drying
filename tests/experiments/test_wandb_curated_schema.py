"""Protect the shared curated W&B history sections and the run organization of both tasks.

Responsibilities:
  - Check that steady and transient runs expose the design's sections, key names and conditional series
  - Check the Optuna keys and which series stay in the run summary
  - Check group, job type and name of steady and transient training runs

Design principles:
  - The design table is the contract: expected key sets are written out, never read back from the producer

This module does NOT:
  - Contact W&B or define model optimization behavior
"""

from __future__ import annotations

from collections import defaultdict

import pytest
from support import configs

from src.experiments import experiments_wandb_schema as schema

_OBJECTIVE = "normalized_drying_group_macro_rmse"
_OVERVIEW = {"objective_validation", "objective_ood", "ood_gap", "train_loss", "learning_rate"}
_PERFORMANCE = {"epoch_seconds", "train_seconds", "samples_per_second", "cuda_peak_memory_gib"}
_TRANSIENT_FIELDS = {"T", "phi", "w_surf", "w_int"}
_PHYSICS_TERMS = {"momentum", "continuity", "boundary"}


def _sections(projections: tuple[schema.MetricProjection, ...]) -> dict[str, set[str]]:
    """Group the destination keys by their section path, e.g. ``Accuracy/OOD``."""
    sections: dict[str, set[str]] = defaultdict(set)
    for projection in projections:
        section, _, name = projection.wandb_key.rpartition("/")
        sections[section].add(name)
    return dict(sections)


def _steady(**overrides: bool) -> tuple[schema.MetricProjection, ...]:
    """Return the steady projections of a physics-informed GPU run with OOD data unless overridden."""
    metrics = configs.resolve(configs.direct_config())["evaluation"]["metrics"]
    flags = {
        "physics_training_enabled": True,
        "physics_monitor_enabled": True,
        "state_aux_enabled": False,
        "ood_enabled": True,
        "cuda_enabled": True,
        "optuna_trial": False,
    } | overrides
    return schema.curated_metric_projections("steady_flow", metrics, objective_id="normalized_group_macro_rmse", **flags)


def _transient(**overrides: bool) -> tuple[schema.MetricProjection, ...]:
    """Return the transient projections of a GPU run with state aux and OOD data unless overridden."""
    flags = {
        "physics_training_enabled": False,
        "physics_monitor_enabled": False,
        "state_aux_enabled": True,
        "ood_enabled": True,
        "cuda_enabled": True,
        "optuna_trial": False,
    } | overrides
    return schema.curated_metric_projections("transient_drying", [], objective_id=_OBJECTIVE, **flags)


def test_steady_history_follows_the_design_sections() -> None:
    """Fill every section for a physics-informed run; OOD accuracy mirrors the validation accuracy names."""
    sections = _sections(_steady())

    assert sections["Overview"] == _OVERVIEW
    assert sections["Loss"] == {"data", "momentum", "continuity", "boundary"}
    assert sections["Physics/Validation"] == {"momentum", "flux_balance", "pressure_boundary"}
    assert sections["Physics/Weight"] == _PHYSICS_TERMS
    assert sections["Performance"] == _PERFORMANCE
    assert sections["Accuracy/Validation"] == sections["Accuracy/OOD"]
    assert {"normalized_rmse_p", "normalized_relative_h1", "physical_velocity_vector_rmse"} <= sections["Accuracy/Validation"]
    assert {"normalized_group_macro_rmse", "normalized_rmse"}.isdisjoint(sections["Accuracy/Validation"])  # objective and multi-field plain RMSE
    assert set(sections) == {"Overview", "Loss", "Physics/Validation", "Physics/Weight", "Performance", "Accuracy/Validation", "Accuracy/OOD"}


def test_transient_history_follows_the_design_sections() -> None:
    """Share Overview, Accuracy and Performance names with steady and add the Curriculum section."""
    sections = _sections(_transient())

    assert sections["Overview"] == _OVERVIEW
    assert sections["Loss"] == {*_TRANSIENT_FIELDS, "state_aux"}
    assert sections["Accuracy/Validation"] == sections["Accuracy/OOD"] == {*_TRANSIENT_FIELDS, "grain_moisture"}
    assert sections["Curriculum"] == {"horizon", "self_fed_stage"}
    assert sections["Performance"] == _PERFORMANCE
    assert set(sections) == {"Overview", "Loss", "Accuracy/Validation", "Accuracy/OOD", "Curriculum", "Performance"}


def test_series_without_a_producer_are_not_listed() -> None:
    """Drop OOD, physics, state aux and CUDA series when the run does not produce them."""
    steady = _sections(_steady(physics_training_enabled=False, physics_monitor_enabled=False, ood_enabled=False, cuda_enabled=False))
    transient = _sections(_transient(state_aux_enabled=False, ood_enabled=False, cuda_enabled=False))

    assert steady["Loss"] == {"data"}
    assert set(steady) == {"Overview", "Loss", "Accuracy/Validation", "Performance"}
    assert steady["Overview"] == {"objective_validation", "train_loss", "learning_rate"}
    assert transient["Loss"] == _TRANSIENT_FIELDS
    assert set(transient) == {"Overview", "Loss", "Accuracy/Validation", "Curriculum", "Performance"}
    assert steady["Performance"] == transient["Performance"] == _PERFORMANCE - {"cuda_peak_memory_gib"}


@pytest.mark.parametrize("task", ["steady", "transient"])
def test_projection_keys_are_unique_and_only_optuna_keys_stay_in_the_summary(task: str) -> None:
    """Map each source and destination once; only the Optuna series keep their last value in the run summary."""
    projections = (_steady if task == "steady" else _transient)(optuna_trial=True)
    sources = [projection.source_key for projection in projections]
    destinations = [projection.wandb_key for projection in projections]
    by_destination = {projection.wandb_key: projection for projection in projections}

    assert len(sources) == len(set(sources))
    assert len(destinations) == len(set(destinations))
    assert _sections(projections)["Optuna"] == {"objective", "best_objective"}
    assert {key for key, projection in by_destination.items() if projection.in_summary} == {"Optuna/objective", "Optuna/best_objective"}
    assert by_destination["Performance/cuda_peak_memory_gib"].multiplier * 1024**3 == 1.0
    assert "Optuna" not in _sections(_steady())
    assert "Optuna" not in _sections(_transient())


def test_producer_source_keys_feed_the_curated_series() -> None:
    """Read the loop's own source names so a series never stays defined but empty."""
    sources = {projection.source_key for projection in _steady()}
    assert {
        "id/normalized_rmse_p",
        "additional_evaluation/normalized_rmse_p",
        "physics/id/flux_balance_mse",
        "physics/train/continuity_weight",
    } <= sources
    assert not any(source.startswith("ood/") for source in sources)


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
            ood_enabled=True,
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
