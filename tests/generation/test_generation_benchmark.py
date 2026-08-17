# ruff: noqa: S101
"""Core-benchmark resources remain outside generated scientific identity."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src import generation

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_COMMIT = "a" * 40


def test_resource_change_preserves_case_identity(
    generation_config_factory: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep scheduler resources operational while generated science stays stable."""
    monkeypatch.setenv("GENERATION_GIT_COMMIT", _COMMIT)
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    case_config = campaign.require_batch(
        material_family="lentil",
        sampling_regime="natural",
    )
    baseline_variant = generation.benchmark.CoreBenchmarkVariant(
        source_path=config_path.parent / "one_core.yaml",
        variant_id="one_core",
        cores_per_case=1,
    )
    suite = generation.benchmark.CoreBenchmarkSuite(
        source_path=config_path.parent / "suite.yaml",
        suite_name="synthetic_core_scaling",
        suite_digest="c" * 64,
        case_campaign_path=config_path,
        case_campaign=campaign,
        case_config=case_config,
        case_index=1,
        repetitions=1,
        variants=(baseline_variant,),
        cores_per_node=24,
        partition="test",
        wall_time=None,
        scheduler_options=(),
        production_campaign_path=config_path,
        production_cores_config_path=config_path.parent / "execution.yaml",
        production_cores_key="cluster.cores_per_case",
    )
    changed_variant = replace(baseline_variant, cores_per_case=2)

    baseline = generation.cases.case.generate_case_input_bundle(
        case_config,
        suite.case_index,
        tmp_path / "baseline",
    )
    changed = generation.cases.case.generate_case_input_bundle(
        case_config,
        suite.case_index,
        tmp_path / "changed-resource",
    )

    assert suite.execution_id(changed_variant) != suite.execution_id(baseline_variant)
    assert changed.case_input_id == baseline.case_input_id
    assert changed.simulation_case_id == baseline.simulation_case_id
    assert changed.case_payload == baseline.case_payload
