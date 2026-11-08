"""Common Generation run planning and lifecycle contracts.

Responsibilities:
  - Check schema-based run dispatch and the authored-campaign plan resolved from a Dataset plan

Design principles:
  - Planning rejects unknown schema kinds before any loader runs

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
