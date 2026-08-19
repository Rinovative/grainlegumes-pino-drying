# ruff: noqa: PLR2004, S101
"""Common Generation run planning and lifecycle contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src import common
from src.generation import generation_run as run_service

COMMIT = "a" * 40


def _write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _campaign(
    *, profile: str = "steady_flow", purpose: str = "family_generalization", packages: tuple[dict[str, object], ...] = ()
) -> SimpleNamespace:
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
        campaign_name=f"{profile}-campaign",
        campaign_digest=f"{profile}-campaign-digest",
        campaign_purpose=purpose,
        batches=(batch,),
        dataset_packages=packages,
        execution_values={"retention_policy": "full" if purpose == "technical_runtime_smoke" else "compact"},
    )


def _benchmark_suite() -> SimpleNamespace:
    canary = SimpleNamespace(variant_id="production", cores_per_case=8)
    other = SimpleNamespace(variant_id="small", cores_per_case=2)
    representatives = (
        SimpleNamespace(case_role="nominal", case_index=3),
        SimpleNamespace(case_role="natural", case_index=4),
    )
    case_config = SimpleNamespace(
        batch_id="benchmark-batch",
        case_input_config_digest="benchmark-input-digest",
        case_id=lambda index: f"case_{index:04d}",
        case_seed=lambda index: 700 + index,
    )
    return SimpleNamespace(
        suite_digest="suite-digest",
        variants=(other, canary),
        representative_case_count=2,
        representative_case=lambda position: representatives[position - 1],
        case_config=case_config,
        canary_variant=lambda: canary,
        variant_wave_order=lambda: (canary, other),
        work_unit_id=lambda variant, position: f"{variant.variant_id}-{representatives[position - 1].case_role}",
    )


@pytest.mark.parametrize(
    ("relative_path", "run_kind", "unit_count"),
    [
        ("configs/generation/workflows/technical_smoke.yaml", "workflow", 0),
        ("configs/generation/benchmarks/transient_core_scaling/suite.yaml", "benchmark", 8),
        ("configs/generation/campaigns/transient_drying/material_pilot.yaml", "campaign", 18),
        ("configs/generation/campaigns/transient_drying/family_generalization.yaml", "campaign", 600),
        ("configs/generation/campaigns/steady_flow/id_dataset.yaml", "campaign", 1050),
    ],
)
def test_every_maintained_entry_config_resolves_one_common_plan(
    relative_path: str,
    run_kind: str,
    unit_count: int,
) -> None:
    """Resolve every documented entry point through schema-kind dispatch."""
    repository = common.paths.get_project_root()

    plan = run_service.resolve_generation_run(
        repository / relative_path,
        source_commit=COMMIT,
        repository_root=repository,
        require_executable=False,
    )

    assert plan.run_kind == run_kind
    assert len(plan.units) == unit_count
    assert plan.config_path == relative_path
    assert plan.to_payload()["schema_version"] == 1


def test_dispatches_by_schema_kind_and_builds_campaign_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve campaign schemas into ordered common work units."""
    config = _write_yaml(tmp_path / "mutable-name.yaml", "schema_kind: generation_campaign\nschema_version: 1\n")
    monkeypatch.setattr(run_service.config_service, "load_campaign_config", lambda *_args, **_kwargs: _campaign())
    monkeypatch.setattr(run_service.campaign_service, "campaign_run_id", lambda *_args, **_kwargs: "durable-campaign-run")

    plan = run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)

    assert plan.run_kind == "campaign"
    assert plan.identity == "durable-campaign-run"
    assert [unit.unit_id for unit in plan.units] == ["steady_flow-batch:case_0000", "steady_flow-batch:case_0001"]
    assert "build_packages" not in plan.lifecycle_stages
    assert plan.to_payload()["config_identity"]


def test_additive_package_request_creates_no_generation_work_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Dataset finalizers independent of the unchanged solver unit plan."""
    config = _write_yaml(
        tmp_path / "campaign.yaml",
        "schema_kind: generation_campaign\nschema_version: 1\n",
    )
    base = _campaign(
        packages=(
            {
                "dataset_name": "transient-id",
                "dataset_view": "transient_drying",
                "evaluation_regime": "id",
            },
        )
    )
    extended = _campaign(
        packages=(
            *base.dataset_packages,
            {
                "dataset_name": "steady-id",
                "dataset_view": "steady_flow",
                "evaluation_regime": "id",
            },
        )
    )
    campaigns = iter((base, extended))
    monkeypatch.setattr(
        run_service.config_service,
        "load_campaign_config",
        lambda *_args, **_kwargs: next(campaigns),
    )
    monkeypatch.setattr(
        run_service.campaign_service,
        "campaign_run_id",
        lambda *_args, **_kwargs: "same-simulation-run",
    )

    base_plan = run_service.resolve_generation_run(
        config,
        source_commit=COMMIT,
        repository_root=tmp_path,
    )
    extended_plan = run_service.resolve_generation_run(
        config,
        source_commit=COMMIT,
        repository_root=tmp_path,
    )

    assert extended_plan.identity == base_plan.identity
    assert extended_plan.dataset_packages == ("transient-id", "steady-id")
    assert [unit.to_payload() for unit in extended_plan.units] == [unit.to_payload() for unit in base_plan.units]
    assert all(unit.unit_kind == "campaign_case" for unit in extended_plan.units)


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


def test_benchmark_units_are_canary_first_and_never_prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan the canary first without inventing a preparation worker unit."""
    config = _write_yaml(
        tmp_path / "suite.yaml",
        "schema_kind: generation_core_scaling_benchmark_suite\nschema_version: 1\n",
    )
    monkeypatch.setattr(run_service.benchmark_service, "load_core_benchmark_suite", lambda *_args, **_kwargs: _benchmark_suite())

    plan = run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)

    assert plan.run_kind == "benchmark"
    assert [unit.unit_id for unit in plan.units] == [
        "production-nominal",
        "production-natural",
        "small-nominal",
        "small-natural",
    ]
    assert all(unit.input_identity not in {"case_0003", "case_0004"} for unit in plan.units)
    assert all(unit.unit_kind == "benchmark_case" for unit in plan.units)
    assert all("prepare" not in unit.unit_kind for unit in plan.units)
    assert plan.units[0].depends_on == ()
    assert plan.units[1].depends_on == ()
    assert plan.units[2].depends_on == ("production-nominal", "production-natural")
    assert plan.units[3].depends_on == ("production-nominal", "production-natural")
    assert all(dict(unit.metadata)["included_in_final_measurements"] for unit in plan.units)
    assert sum(bool(dict(unit.metadata)["canary"]) for unit in plan.units) == 2
    assert "build_packages" not in plan.lifecycle_stages


def test_workflow_preserves_child_order_and_binds_authored_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind ordered Smoke children and the paired finalizer into one plan."""
    steady = _write_yaml(tmp_path / "steady.yaml", "schema_kind: generation_campaign\nschema_version: 1\n")
    transient = _write_yaml(tmp_path / "transient.yaml", "schema_kind: generation_campaign\nschema_version: 1\n")
    workflow_directory = tmp_path / "configs" / "generation" / "workflows"
    workflow_directory.mkdir(parents=True)
    workflow = _write_yaml(
        workflow_directory / "smoke.yaml",
        """schema_kind: generation_workflow
schema_version: 1
workflow_name: paired-smoke
children:
  - steady.yaml
  - transient.yaml
finalizers:
  - paired_technical_smoke
""",
    )
    by_path = {
        steady.resolve(): _campaign(profile="steady_flow", purpose="technical_runtime_smoke"),
        transient.resolve(): _campaign(profile="transient_drying", purpose="technical_runtime_smoke"),
    }
    monkeypatch.setattr(run_service.config_service, "load_campaign_config", lambda path, **_kwargs: by_path[Path(path).resolve()])

    plan = run_service.resolve_generation_run(workflow, source_commit=COMMIT, repository_root=tmp_path)

    assert plan.run_kind == "workflow"
    assert [child.input_identity for child in plan.children] == ["steady_flow-campaign-digest", "transient_drying-campaign-digest"]
    assert plan.finalizers[0].finalizer_kind == "paired_technical_smoke"
    assert plan.identity.startswith("workflow__")


def test_controller_selects_common_lifecycle_and_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance only the next declared stage and stop before deferred transfer."""
    config = _write_yaml(tmp_path / "campaign.yaml", "schema_kind: generation_campaign\nschema_version: 1\n")
    packages: tuple[dict[str, object], ...] = ({"dataset_name": "airflow-id"},)
    monkeypatch.setattr(run_service.config_service, "load_campaign_config", lambda *_args, **_kwargs: _campaign(packages=packages))
    plan = run_service.resolve_generation_run(config, source_commit=COMMIT, repository_root=tmp_path)

    controller = run_service.GenerationRunController(plan)
    assert controller.next_stage() == "resolve_config"
    with pytest.raises(ValueError, match="expected 'resolve_config'"):
        controller.advance("submit")
    controller = controller.advance("resolve_config")
    assert controller.continuation_state() == "preflight_ready"
    deferred = controller.resume(defer_collection=True)
    assert deferred.next_stage() == "preflight"
    transfer_index = plan.lifecycle_stages.index("transfer")
    completed = frozenset(plan.lifecycle_stages[:transfer_index])
    awaiting = run_service.GenerationRunController(
        plan,
        completed_stages=completed,
        defer_collection=True,
    )
    assert awaiting.next_stage() is None
    assert awaiting.continuation_state() == "awaiting_collection"


def test_workflow_rejects_duplicate_and_unsafe_children(tmp_path: Path) -> None:
    """Reject duplicate or repository-external workflow children."""
    _write_yaml(tmp_path / "child.yaml", "schema_kind: generation_campaign\nschema_version: 1\n")
    duplicate = _write_yaml(
        tmp_path / "duplicate.yaml",
        """schema_kind: generation_workflow
schema_version: 1
workflow_name: paired-smoke
children: [child.yaml, child.yaml]
finalizers: [paired_technical_smoke]
""",
    )
    with pytest.raises(ValueError, match="duplicate-free"):
        run_service.resolve_generation_run(duplicate, source_commit=COMMIT, repository_root=tmp_path)

    unsafe = _write_yaml(
        tmp_path / "unsafe.yaml",
        """schema_kind: generation_workflow
schema_version: 1
workflow_name: paired-smoke
children: [/tmp/outside.yaml, child.yaml]
finalizers: [paired_technical_smoke]
""",
    )
    with pytest.raises(ValueError, match="must not use an absolute path"):
        run_service.resolve_generation_run(unsafe, source_commit=COMMIT, repository_root=tmp_path)
