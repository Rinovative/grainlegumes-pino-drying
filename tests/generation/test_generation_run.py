"""Common Generation run planning and lifecycle contracts.

Responsibilities:
  - Check schema-based run dispatch, benchmark order, and byte-bound workflow children

Design principles:
  - Planning preserves authored child order and rejects unsafe duplicate membership

This module does NOT:
  - Prepare cases or submit scheduler jobs during planning
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from src import common
from src.generation import generation_run as run_service
from src.generation.cases import generation_cases_config as config_service
from src.generation.cli import cli_generation

COMMIT = "a" * 40


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _campaign(*, profile: str = "steady_flow", purpose: str = "source_generation") -> SimpleNamespace:
    batch = SimpleNamespace(
        batch_id=f"{profile}-batch",
        case_indices=(0, 1),
        material_family="Lentil",
        sampling_regime="natural",
        profile=SimpleNamespace(id=profile),
        case_input_config_digest="input-digest",
        case_id=lambda index: f"case_{index:04d}",
        case_seed=lambda index: 100 + index,
    )
    return SimpleNamespace(
        schema_version=1,
        profile=SimpleNamespace(id=profile),
        campaign_name=f"{profile}-campaign",
        campaign_digest=f"{profile}-campaign-digest",
        campaign_purpose=purpose,
        scientific_overrides={},
        execution_overrides={},
        authored_value=None,
        batches=(batch,),
        execution_values={"retention_policy": "compact"},
    )


def _benchmark_suite() -> SimpleNamespace:
    small = SimpleNamespace(variant_id="small", cores_per_case=2)
    large = SimpleNamespace(variant_id="large", cores_per_case=8)
    sweeps = (SimpleNamespace(sweep_id="sweep_1", order="ascending"), SimpleNamespace(sweep_id="sweep_2", order="descending"))
    representatives = (
        SimpleNamespace(case_role="case_01", case_index=3),
        SimpleNamespace(case_role="case_02", case_index=4),
    )
    case_config = SimpleNamespace(
        batch_id="benchmark-batch",
        case_input_config_digest="benchmark-input-digest",
        case_id=lambda index: f"case_{index:04d}",
        case_seed=lambda index: 700 + index,
    )
    return SimpleNamespace(
        suite_digest="suite-digest",
        variants=(small, large),
        sweeps=sweeps,
        sweep_variants=lambda sweep: (small, large) if sweep.order == "ascending" else (large, small),
        resources=lambda variant: SimpleNamespace(concurrent_cases=32 // variant.cores_per_case),
        representative_case_count=2,
        target_successful_cases=1,
        representative_case=lambda position: representatives[position - 1],
        case_config=case_config,
        work_unit_id=lambda sweep, variant, attempt, position: (
            f"{sweep.sweep_id}-{variant.variant_id}-a{attempt}-{representatives[position - 1].case_role}"
        ),
    )


def test_dispatches_by_schema_kind_and_builds_campaign_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve campaign schemas into ordered common work units."""
    config = _write_yaml(tmp_path / "mutable-name.yaml", "schema_kind: generation_source_campaign\nschema_version: 1\n")
    monkeypatch.setattr(run_service.config_service, "load_campaign_config", lambda *_args, **_kwargs: _campaign())
    monkeypatch.setattr(run_service.campaign_service, "campaign_run_id", lambda *_args, **_kwargs: "durable-campaign-run")

    plan = run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)

    assert plan.run_kind == "campaign"
    assert plan.identity == "durable-campaign-run"
    assert [unit.unit_id for unit in plan.units] == ["steady_flow-batch:case_0000", "steady_flow-batch:case_0001"]
    assert "build_packages" not in plan.lifecycle_stages
    assert plan.to_payload()["config_identity"]


def test_rejects_unknown_top_level_schema_without_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown schema kinds before invoking a specialized loader."""
    config = _write_yaml(tmp_path / "anything.yaml", "schema_kind: unknown\nschema_version: 1\n")
    monkeypatch.setattr(
        run_service.config_service,
        "load_campaign_config",
        lambda *_args, **_kwargs: pytest.fail("campaign loader must not be called"),
    )

    with pytest.raises(ValueError, match="Unsupported Generation run schema_kind"):
        run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)


def test_benchmark_refuses_a_source_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A revision relabels a Source campaign; a benchmark has no Source, so the override is refused before its suite loads."""
    config = _write_yaml(tmp_path / "suite.yaml", "schema_kind: generation_core_scaling_benchmark_suite\nschema_version: 1\n")
    monkeypatch.setattr(run_service.benchmark_config, "load_core_benchmark_suite", lambda *_args, **_kwargs: pytest.fail("suite must not load"))

    with pytest.raises(ValueError, match="source campaigns only"):
        run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path, scientific_overrides={"revision": "fix"})


def test_benchmark_units_follow_sweep_groups_and_never_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan each sweep's variant groups in sweep order, each group after the previous one, without a preparation unit."""
    config = _write_yaml(
        tmp_path / "suite.yaml",
        "schema_kind: generation_core_scaling_benchmark_suite\nschema_version: 3\n",
    )
    monkeypatch.setattr(run_service.benchmark_config, "load_core_benchmark_suite", lambda *_args, **_kwargs: _benchmark_suite())

    plan = run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)

    assert plan.run_kind == "benchmark"
    assert [unit.unit_id for unit in plan.units] == [
        "sweep_1-small-a1-case_01",
        "sweep_1-small-a1-case_02",
        "sweep_1-large-a1-case_01",
        "sweep_1-large-a1-case_02",
        "sweep_2-large-a1-case_01",
        "sweep_2-large-a1-case_02",
        "sweep_2-small-a1-case_01",
        "sweep_2-small-a1-case_02",
    ]
    assert all(unit.unit_kind == "benchmark_case" for unit in plan.units)
    assert plan.units[0].depends_on == ()
    assert plan.units[2].depends_on == ("sweep_1-small-a1-case_01", "sweep_1-small-a1-case_02")
    assert plan.units[4].depends_on == ()
    assert plan.units[6].depends_on == ("sweep_2-large-a1-case_01", "sweep_2-large-a1-case_02")
    assert dict(plan.units[2].metadata)["concurrent_cases"] == 4
    assert all(dict(unit.metadata)["success_target"] == 1 for unit in plan.units)
    assert "build_packages" not in plan.lifecycle_stages


def test_authored_campaign_plan_dispatches_mapping_and_preserves_plan_bytes(
    generation_config_factory: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Actual command parsing resolves an authored campaign from a Dataset plan."""
    path, _ = generation_config_factory(simulation_profile="steady_flow", natural_count=2)
    authored = yaml.safe_load(path.read_text())
    for key in ("sources_config", "registry_config", "common_config", "operations_config", "profile_config", "execution_config"):
        authored[key] = str((path.parent / authored[key]).resolve())
    plan_path = common.paths.get_project_root() / "configs/02_datasets/test-plan.yaml"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text("schema_kind: dataset_plan\nroles: {lentil: train}\n")
    original = plan_path.read_bytes()
    expected = config_service.load_campaign_config(plan_path, authored_value=authored)
    assert (
        cli_generation.main(
            [
                "resolve-generation-run",
                str(plan_path),
                "--git-commit",
                COMMIT,
                "--authored-value-json",
                json.dumps(authored),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["authored_value"] == authored
    assert payload["input_identity"] == expected.campaign_digest
    assert payload["config_path"] == "configs/02_datasets/test-plan.yaml"
    assert cli_generation.main(["validate-config", str(plan_path), "--authored-value-json", json.dumps(authored)]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["total_case_count"] == 2
    assert plan_path.read_bytes() == original
