# ruff: noqa: PLR2004, S101
"""Protect read-only persisted Evaluation run discovery."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import pytest
import yaml

from src import common
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.analysis.evaluation import evaluation_run_discovery as discovery
from src.analysis.evaluation import evaluation_selection as selection
from src.analysis.evaluation import evaluation_workspace as workspace
from src.experiments import experiments_run_identity as identity


def _write_run(
    root: Path,
    *,
    leaf: str,
    task: str = "transient_drying",
    run_name: str | None = None,
    status: str = "completed",
    updated_at: str = "2026-01-01T00:00:00+00:00",
    stage: str | None = None,
    comparison_arm: str | None = None,
    artifact: bool = False,
    evaluation: dict[str, Any] | None = None,
) -> Path:
    """Write one compact persisted leaf without loading project services."""
    run_dir = root / task / "runs" / leaf
    run_dir.mkdir(parents=True)
    name = run_name or f"scientific-{leaf}"
    config: dict[str, Any] = {
        "task": task,
        "run": {"name": name, "seed": 17},
        "model": {"kind": "fno"},
        "data": {"train_dataset": "dataset-id", "ood_datasets": ["dataset-ood"]},
    }
    if evaluation is not None:
        config["evaluation"] = evaluation
    if stage is not None or comparison_arm is not None:
        config["training"] = {}
        if stage is not None:
            config["training"]["stage"] = stage
        if comparison_arm is not None:
            config["training"]["comparison_arm"] = comparison_arm
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": task,
                "run_name": name,
                "status": status,
                "updated_at": updated_at,
                "effective_config_digest": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best_checkpoint.pt").write_bytes(b"best")
    if artifact:
        for artifact_root in (run_dir / "analysis" / "id", run_dir / "analysis" / "ood" / "dataset-ood"):
            artifact_root.mkdir(parents=True)
            (artifact_root / "cases.parquet").write_bytes(b"parquet")
            payload_root = artifact_root / "npz"
            payload_root.mkdir()
            (payload_root / "case-0.npz").write_bytes(b"npz")
            marker = {
                "provenance_schema_version": artifact_contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
                "artifact_schema_version": artifact_contracts.ARTIFACT_SCHEMA_VERSION,
                "run": {"task": task},
                "outputs": artifact_contracts.artifact_output_manifest(artifact_root),
            }
            (artifact_root / "artifact_provenance.json").write_text(json.dumps(marker), encoding="utf-8")
    return run_dir


def _write_scoped_artifact(
    root: Path,
    *,
    run_dir: Path,
    role: str = "id",
) -> Path:
    """Write one compact marker-complete selected-case discovery fixture."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    scoped = root / "transient_drying" / "scoped_artifacts" / "one-case"
    scoped.mkdir(parents=True)
    (scoped / "cases.parquet").write_bytes(b"parquet")
    payload_root = scoped / "npz"
    payload_root.mkdir()
    (payload_root / "case-0.npz").write_bytes(b"npz")
    selected = ["transient_drying__chickpea__natural__case_0001"]
    marker = {
        "provenance_schema_version": artifact_contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": discovery.transient_artifact.TRANSIENT_SEQUENCE_SCHEMA_VERSION,
        "artifact_kind": discovery.transient_artifact.TRANSIENT_ARTIFACT_KIND,
        "task": "transient_drying",
        "run": {
            "name": config["run"]["name"],
            "effective_config_digest": summary["effective_config_digest"],
            "best_checkpoint_sha256": common.serialization.file_sha256(run_dir / "best_checkpoint.pt"),
        },
        "dataset": {
            "name": config["data"]["train_dataset"],
            "role": role,
        },
        "evaluation": {
            "scope": {
                "kind": "selected_cases",
                "canonical_publication_eligible": False,
                "selected_case_ids": selected,
                "saved_case_ids_by_source": [selected],
            }
        },
        "outputs": artifact_contracts.artifact_output_manifest(scoped),
    }
    (scoped / artifact_contracts.ARTIFACT_PROVENANCE_FILENAME).write_text(
        json.dumps(marker),
        encoding="utf-8",
    )
    return scoped


def _write_parent(root: Path, *, label: str, children: tuple[Path, Path]) -> str:
    """Write one validator-admitted transient parent record for two exact leaves."""
    payload = {
        "schema_kind": identity.EXPERIMENT_RECORD_SCHEMA_KIND,
        "schema_version": identity.EXPERIMENT_RECORD_SCHEMA_VERSION,
        "task": "transient_drying",
        "parent_label": label,
        "run_revision": 0,
        "seed": 17,
        "authored_config": {"path": "synthetic.yaml", "basename": "synthetic.yaml", "sha256": "a" * 64},
        "dataset_identity": {"train_dataset": "dataset-id", "ood_datasets": ["dataset-ood"], "references": None},
        "children": {
            "stage_a0": {"run_name": "scientific-a", "path": str(children[0]), "resolved_config_sha256": "a" * 64},
            "stage_b": {"run_name": "scientific-b", "path": str(children[1]), "resolved_config_sha256": "b" * 64},
        },
        "handoff": {"source_run_name": "scientific-a", "target_run_name": "scientific-b"},
    }
    record = {
        **payload,
        "parent_identity_sha256": common.serialization.canonical_json_sha256(payload),
        "source_repository": {"commit": None, "dirty": None},
    }
    path = root / "transient_drying" / "runs" / label / "experiment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record["parent_identity_sha256"]


def _widget_descriptions(widget: widgets.Widget) -> tuple[str, ...]:
    """Return rendered widget descriptions recursively."""
    values: list[str] = []
    description = getattr(widget, "description", None)
    if isinstance(description, str) and description:
        values.append(description)
    for child in getattr(widget, "children", ()):
        values.extend(_widget_descriptions(child))
    return tuple(values)


def test_dataset_display_projection_is_material_first_and_digest_free() -> None:
    """Keep exact Dataset identity secondary to concise material/role text."""
    id_dataset = "transient_drying__lentil+chickpea__id__edeafe22ec484ec3"
    ood_dataset = "transient_drying__kidney_bean__near_family_ood__3cc7db648cf437b5"

    assert (
        discovery.dataset_display_projection(
            "transient_drying",
            id_dataset,
        )
        == "lentil+chickpea_id"
    )
    assert (
        workspace._EvaluationWorkspaceController._dataset_binding_label(  # noqa: SLF001
            task="transient_drying",
            dataset_name=id_dataset,
            role="id",
        )
        == "Lentil + Chickpea · ID"
    )
    assert (
        workspace._EvaluationWorkspaceController._dataset_binding_label(  # noqa: SLF001
            task="transient_drying",
            dataset_name=ood_dataset,
            role="ood",
        )
        == "Kidney bean · Near-family OOD"
    )


def test_discovers_current_group_and_legacy_leaf_with_exact_identity(tmp_path: Path) -> None:
    """Group only record-declared transient children without changing a legacy leaf."""
    stage_a = _write_run(tmp_path, leaf="storage-a", run_name="scientific-a", stage="a")
    stage_b = _write_run(tmp_path, leaf="storage-b", run_name="scientific-b", stage="b")
    legacy = _write_run(tmp_path, leaf="opaque-legacy", task="steady_flow", run_name="exact-steady")
    parent_identity = _write_parent(tmp_path, label="concise-parent", children=(stage_a, stage_b))

    catalog = discovery.discover_evaluation_runs(tmp_path)

    assert len(catalog.runs) == len({"scientific-a", "scientific-b", "exact-steady"})
    assert len(catalog.groups) == len({"current-parent", "legacy-leaf"})
    grouped = next(group for group in catalog.groups if group.identity_sha256 == parent_identity)
    assert {item.run_name for item in grouped.children} == {"scientific-a", "scientific-b"}
    legacy_record = next(item for item in catalog.runs if item.run_name == "exact-steady")
    assert legacy_record.run_dir == legacy.resolve()
    assert legacy_record.parent_identity_sha256 is None
    assert legacy_record.dataset_id == "dataset-id"
    assert legacy_record.identity_format == "legacy"
    assert len(next(group for group in catalog.groups if legacy_record in group.children).children) == 1


def test_failed_runs_remain_visible_and_artifacts_report_missing_or_available(
    tmp_path: Path,
) -> None:
    """Keep unavailable lifecycle evidence visible while inspecting artifacts read-only."""
    failed = _write_run(tmp_path, leaf="failed", status="failed")
    available = _write_run(tmp_path, leaf="available", artifact=True)

    catalog = discovery.discover_evaluation_runs(tmp_path)

    failed_record = next(item for item in catalog.runs if item.run_dir == failed.resolve())
    available_record = next(item for item in catalog.runs if item.run_dir == available.resolve())
    assert failed_record.availability == "unavailable"
    assert failed_record.id_artifact.state == "missing"
    assert available_record.id_artifact.state == "ready"
    assert {"aggregate", "case_fields"}.issubset(available_record.id_artifact.capabilities)
    assert available_record.availability == "available"
    assert available_record.evaluable
    assert available_record.action_enabled
    assert available_record.artifact_command == f"./scripts/docker_job.sh artifacts --run-dir {available.resolve()}"


def test_marker_discovery_does_not_hash_numerical_payloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep catalog discovery metadata-only and defer hashes to selected admission."""
    run_dir = _write_run(tmp_path, leaf="available", artifact=True)

    def unexpected_hash(_root: Path) -> object:
        pytest.fail("run discovery rehashed artifact payloads")

    monkeypatch.setattr(
        discovery.artifact_contracts,
        "artifact_output_manifest",
        unexpected_hash,
    )

    catalog = discovery.discover_evaluation_runs(tmp_path)

    record = next(item for item in catalog.runs if item.run_dir == run_dir.resolve())
    assert record.id_artifact.state == "ready"
    assert record.ood_artifact.state == "ready"
    assert record.id_artifact.digest is not None


def test_ordering_and_dispatch_come_from_persisted_metadata(tmp_path: Path) -> None:
    """Prefer completed newer leaves and never dispatch from directory aliases."""
    _write_run(tmp_path, leaf="newer-alias", task="steady_flow", run_name="persisted-steady", updated_at="2026-02-01T00:00:00+00:00")
    _write_run(tmp_path, leaf="older-alias", task="transient_drying", run_name="persisted-transient", updated_at="2026-01-01T00:00:00+00:00")

    catalog = discovery.discover_evaluation_runs(tmp_path)

    assert [item.run_name for item in catalog.runs] == ["persisted-steady", "persisted-transient"]
    assert [item.task for item in catalog.runs] == ["steady_flow", "transient_drying"]
    assert catalog.counts_by_task == {"steady_flow": 1, "transient_drying": 1}
    transient = next(item for item in catalog.runs if item.task == "transient_drying")
    assert transient.stage is None


def test_selection_state_synchronizes_cross_experiment_task_and_artifact_coordinates(
    tmp_path: Path,
) -> None:
    """Keep one observable owner for runs, scope, channels, case, and exact time."""
    first = _write_run(tmp_path, leaf="transient-a", run_name="transient-a")
    second = _write_run(tmp_path, leaf="transient-b", run_name="transient-b")
    steady = _write_run(
        tmp_path,
        leaf="steady",
        task="steady_flow",
        run_name="steady",
    )
    catalog = discovery.discover_evaluation_runs(tmp_path)
    state = selection.EvaluationSelectionState(catalog, comparison=True)
    changes: list[frozenset[str]] = []
    state.observe(lambda _before, _after, changed: changes.append(changed))

    state.select_catalog_runs((first, second))
    state.select_scope("single")
    state.bind_capabilities(
        selection.EvaluationViewCapabilities(
            task="transient_drying",
            channels=("T", "phi", "w_surf", "w_int"),
            case_ids=("case-a", "case-b"),
            physical_times=(0.0, 1.5, 3.0),
            protocols=("teacher_forced_one_step", "autonomous_full"),
            horizons=(1, "full"),
        )
    )

    assert state.selection.task == "transient_drying"
    assert len(state.selection.experiment_identities) == 2
    assert state.selection.run_dirs == (first.resolve(), second.resolve())
    assert state.selection.scope == "single"
    assert state.selection.case_id == "case-a"
    assert state.selection.physical_time == 3.0
    assert state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert any("run_dirs" in changed for changed in changes)
    with pytest.raises(ValueError, match="unavailable"):
        state.select_physical_time(2.0)

    steady_group = next(group for group in catalog.groups if any(run.run_dir == steady.resolve() for run in group.children))
    state.select_experiment(steady_group.identity_sha256)
    state.bind_capabilities(
        selection.EvaluationViewCapabilities(
            task="steady_flow",
            channels=("p", "u_x", "u_y"),
            case_ids=("steady-case",),
        )
    )
    assert state.selection.physical_time is None
    assert state.selection.protocol is None
    assert state.selection.horizon is None
    with pytest.raises(ValueError, match="Steady Evaluation"):
        selection.EvaluationViewCapabilities(
            task="steady_flow",
            channels=("p",),
            case_ids=("case",),
            physical_times=(0.0,),
        )


def test_discovery_reports_generating_only_for_active_staging_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish active private staging from stale invalid evidence."""
    run_dir = _write_run(tmp_path, leaf="generating")
    target = run_dir / "analysis" / "id"
    stage = target.parent / ".id.staging.active"
    stage.mkdir(parents=True)
    monkeypatch.setattr(
        discovery,
        "_artifact_writer_active",
        lambda root: root == target,
    )

    active = discovery.discover_evaluation_runs(tmp_path).runs[0]
    assert active.id_artifact.state == "generating"
    assert "aggregate" not in active.id_artifact.capabilities

    monkeypatch.setattr(discovery, "_artifact_writer_active", lambda _root: False)
    stale = discovery.discover_evaluation_runs(tmp_path).runs[0]
    assert stale.id_artifact.state == "invalid"


def test_canonical_ready_precedes_failed_staging_and_scoped_partial(
    tmp_path: Path,
) -> None:
    """Keep complete canonical evidence authoritative over lower lifecycle states."""
    run_dir = _write_run(
        tmp_path,
        leaf="canonical",
        run_name="scientific-canonical",
        artifact=True,
    )
    failed_stage = run_dir / "analysis" / ".id.staging.failed"
    failed_stage.mkdir()
    _write_scoped_artifact(tmp_path, run_dir=run_dir)

    record = discovery.discover_evaluation_runs(tmp_path).runs[0]

    assert record.id_artifact.state == "ready"
    assert record.id_artifact.root == run_dir / "analysis" / "id"


def test_run_overview_uses_material_first_roles_and_shared_protocol_labels(
    tmp_path: Path,
) -> None:
    """Present authoritative material coverage before run-relative role metadata."""
    _write_run(
        tmp_path,
        leaf="overview",
        artifact=True,
        evaluation={
            "objective": {"id": "normalized_drying_group_macro_rmse"},
        },
    )
    record = replace(
        discovery.discover_evaluation_runs(tmp_path).runs[0],
        spatial_stride=1,
    )
    controller = workspace._EvaluationWorkspaceController.__new__(  # noqa: SLF001
        workspace._EvaluationWorkspaceController  # noqa: SLF001
    )
    html = controller._status_html(  # noqa: SLF001
        record,
        selected_roles=("id", "ood"),
        material_coverage=(
            ("lentil", "id"),
            ("chickpea", "id"),
            ("kidney_bean", "ood"),
        ),
    )

    assert "Lentil · ID · available" in html
    assert "Chickpea · ID · available" in html
    assert "Kidney bean · Near-family OOD · available" in html
    assert "Drying group macro RMSE" in html
    assert "Spatial stride</th><td>1" in html
    assert "Generate full artifact coverage" not in html


def test_scoped_partial_is_exactly_associated_and_single_case_gated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover one exact scoped artifact and pass it only to Single-case Evaluation."""
    run_dir = _write_run(tmp_path, leaf="scoped", run_name="scientific-scoped")
    scoped = _write_scoped_artifact(tmp_path, run_dir=run_dir)
    catalog = discovery.discover_evaluation_runs(tmp_path)
    record = catalog.runs[0]

    assert record.id_artifact.state == "scoped_partial"
    assert record.id_artifact.root == scoped
    assert record.id_artifact.case_count == 1
    assert "aggregate" not in record.id_artifact.capabilities
    assert {"case_fields", "trajectory", "rollout"}.issubset(record.id_artifact.capabilities)

    calls: list[dict[str, object]] = []

    class FakeWorkflow:
        def __init__(self) -> None:
            self.panel = widgets.Label("scoped")

        def close(self) -> None:
            pass

    def prepare(_run_dir: Path, **kwargs: object) -> FakeWorkflow:
        calls.append(dict(kwargs))
        return FakeWorkflow()

    monkeypatch.setattr(
        workspace.workflow,
        "prepare_single_model_evaluation_workspace",
        prepare,
    )
    prepared = workspace.prepare_single_model_evaluation_workspace(
        experiments_root=tmp_path,
    )
    controller = prepared._controller  # noqa: SLF001
    assert controller is not None
    assert calls[-1]["artifact_roots"] == {"id": scoped}
    assert calls[-1]["sections"] == ("sample_viewer",)
    assert "ID · scoped partial (1 case)" in controller.status.value
    assert "Generate full artifact coverage" in controller.status.value
    assert "--one-case" not in controller.status.value
    prepared.close()


def test_workspace_discovers_once_and_renders_missing_artifacts_without_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build both notebook modes from one scan and keep unavailable runs visible."""
    experiments_root = tmp_path / "03_experiments"
    _write_run(
        experiments_root,
        leaf="missing-artifact",
        run_name="persisted-transient",
    )
    scans: list[Path] = []
    original = discovery.discover_evaluation_runs

    def counted(root: Path | str) -> discovery.EvaluationRunCatalog:
        scans.append(Path(root))
        return original(root)

    monkeypatch.setattr(workspace.run_discovery, "discover_evaluation_runs", counted)
    monkeypatch.setattr(
        workspace.workflow,
        "prepare_single_model_evaluation_workspace",
        lambda *_args, **_kwargs: pytest.fail("missing artifact triggered inference/loading"),
    )

    prepared = workspace.prepare_single_model_evaluation_workspace(experiments_root=experiments_root)
    assert len(scans) == 1
    assert prepared.selection_state is not None
    assert prepared.catalog.counts_by_artifact_state["missing"] == 2
    assert "Artifact roles: missing=2" in prepared.summary_text
    controller = prepared._controller  # noqa: SLF001
    assert controller is not None
    controller.run.value = controller.run.value
    assert len(scans) == 1
    descriptions = set(_widget_descriptions(controller.panel))
    assert descriptions.isdisjoint({"Task:", "Stage:", "Seed:", "Regime:", "View:"})
    assert descriptions == {"Run:"}
    assert "ID · missing; Near-family OOD · missing" in controller.status.value
    assert "docker_job.sh" in controller.status.value
    assert "--queue-gpu" not in controller.status.value
    assert "--one-case" not in controller.status.value
    prepared.close()

    compared = workspace.prepare_model_comparison_evaluation_workspace(experiments_root=experiments_root)
    assert len(scans) == 2
    comparison_controller = compared._controller  # noqa: SLF001
    assert comparison_controller is not None
    assert "Select at least two runs" in comparison_controller.status.value
    compared.close()


def test_comparison_workspace_admits_exact_a0_a_plus_b_stage_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the workflow validate one exact three-arm transient comparison."""
    experiments_root = tmp_path / "03_experiments"
    run_dirs = tuple(
        _write_run(
            experiments_root,
            leaf=arm,
            run_name=f"scientific-{arm}",
            comparison_arm=arm,
            artifact=True,
        )
        for arm in ("a0", "a_plus", "b")
    )
    calls: list[tuple[Path, ...]] = []

    class FakeWorkflow:
        def __init__(self) -> None:
            self.panel = widgets.Label("three-arm")
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def prepare(selected: tuple[Path, ...], **kwargs: object) -> FakeWorkflow:
        assert kwargs["auto_build_missing"] is False
        calls.append(tuple(selected))
        return FakeWorkflow()

    monkeypatch.setattr(
        workspace.workflow,
        "prepare_model_comparison_evaluation_workspace",
        prepare,
    )
    prepared = workspace.prepare_model_comparison_evaluation_workspace(
        experiments_root=experiments_root,
    )
    controller = prepared._controller  # noqa: SLF001
    assert controller is not None

    controller.run_filter.value = tuple(str(path.resolve()) for path in run_dirs)

    assert calls[-1] == tuple(path.resolve() for path in run_dirs)
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.run_dirs == calls[-1]
    assert "incompatible stage" not in controller.status.value
    prepared.close()


def test_single_workspace_switches_transient_steady_transient_without_state_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild task-owned workflows from run metadata within one live notebook."""
    experiments_root = tmp_path / "03_experiments"
    transient_run = _write_run(
        experiments_root,
        leaf="switch-transient",
        task="transient_drying",
        artifact=True,
    )
    steady_run = _write_run(
        experiments_root,
        leaf="switch-steady",
        task="steady_flow",
        artifact=True,
    )
    calls: list[tuple[Path, str]] = []

    class FakeWorkflow:
        def __init__(self, task: str) -> None:
            self.task = task
            self.panel = widgets.Label(f"{task}-panel")
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    def prepare(run_dir: Path, **kwargs: Any) -> FakeWorkflow:
        task = kwargs["expected_task"]
        assert task in {"steady_flow", "transient_drying"}
        state = kwargs["selection_state"]
        capabilities = (
            selection.EvaluationViewCapabilities(
                task="steady_flow",
                channels=("p", "u", "v"),
                case_ids=("steady-case",),
            )
            if task == "steady_flow"
            else selection.EvaluationViewCapabilities(
                task="transient_drying",
                channels=("T", "phi", "w_surf", "w_int"),
                case_ids=("transient-case",),
                physical_times=(0.0, 1.0),
                protocols=("autonomous_full",),
                horizons=("full",),
            )
        )
        state.bind_capabilities(capabilities)
        calls.append((Path(run_dir).resolve(), task))
        return FakeWorkflow(task)

    monkeypatch.setattr(
        workspace.workflow,
        "prepare_single_model_evaluation_workspace",
        prepare,
    )
    prepared = workspace.prepare_single_model_evaluation_workspace(
        experiments_root=experiments_root,
    )
    controller = prepared._controller  # noqa: SLF001
    assert controller is not None

    controller.run.value = str(transient_run.resolve())
    calls.clear()
    controller._render_single()  # noqa: SLF001
    transient_first = controller._workflow  # noqa: SLF001
    assert isinstance(transient_first, FakeWorkflow)
    assert calls == [(transient_run.resolve(), "transient_drying")]
    assert controller.analysis.children == (transient_first.panel,)
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.task == "transient_drying"
    assert prepared.selection_state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert prepared.selection_state.selection.physical_time == 1.0
    assert prepared.selection_state.selection.protocol == "autonomous_full"
    assert prepared.selection_state.selection.horizon == "full"

    controller.run.value = str(steady_run.resolve())
    steady = controller._workflow  # noqa: SLF001
    assert isinstance(steady, FakeWorkflow)
    assert transient_first.close_count == 1
    assert controller.analysis.children == (steady.panel,)
    assert prepared.selection_state.selection.task == "steady_flow"
    assert prepared.selection_state.selection.channels == ("p", "u", "v")
    assert prepared.selection_state.selection.physical_time is None
    assert prepared.selection_state.selection.protocol is None
    assert prepared.selection_state.selection.horizon is None

    controller.run.value = str(transient_run.resolve())
    transient_second = controller._workflow  # noqa: SLF001
    assert isinstance(transient_second, FakeWorkflow)
    assert transient_second is not transient_first
    assert steady.close_count == 1
    assert controller.analysis.children == (transient_second.panel,)
    assert prepared.selection_state.selection.task == "transient_drying"
    assert prepared.selection_state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert prepared.selection_state.selection.physical_time == 1.0
    assert tuple(task for _run_dir, task in calls) == (
        "transient_drying",
        "steady_flow",
        "transient_drying",
    )
    prepared.close()
    assert transient_second.close_count == 1


def test_comparison_workspace_switches_tasks_and_rejects_mixed_task_hybrids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch compatible task pairs separately and clear mixed-task panels."""
    experiments_root = tmp_path / "03_experiments"
    steady_runs = tuple(
        _write_run(
            experiments_root,
            leaf=f"comparison-steady-{index}",
            task="steady_flow",
            artifact=True,
        )
        for index in range(2)
    )
    transient_runs = tuple(
        _write_run(
            experiments_root,
            leaf=f"comparison-transient-{index}",
            task="transient_drying",
            artifact=True,
        )
        for index in range(2)
    )
    calls: list[tuple[tuple[Path, ...], str]] = []

    class FakeWorkflow:
        def __init__(self, task: str) -> None:
            self.task = task
            self.panel = widgets.Label(f"{task}-comparison-panel")
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    def prepare(run_dirs: tuple[Path, ...], **kwargs: Any) -> FakeWorkflow:
        task = kwargs["expected_task"]
        state = kwargs["selection_state"]
        capabilities = (
            selection.EvaluationViewCapabilities(
                task="steady_flow",
                channels=("p", "u", "v"),
                case_ids=("shared-steady-case",),
            )
            if task == "steady_flow"
            else selection.EvaluationViewCapabilities(
                task="transient_drying",
                channels=("T", "phi", "w_surf", "w_int"),
                case_ids=("shared-transient-case",),
                physical_times=(0.0, 1.0),
                protocols=("autonomous_full",),
                horizons=("full",),
            )
        )
        state.bind_capabilities(capabilities)
        resolved = tuple(Path(run_dir).resolve() for run_dir in run_dirs)
        calls.append((resolved, task))
        return FakeWorkflow(task)

    monkeypatch.setattr(
        workspace.workflow,
        "prepare_model_comparison_evaluation_workspace",
        prepare,
    )
    prepared = workspace.prepare_model_comparison_evaluation_workspace(
        experiments_root=experiments_root,
    )
    controller = prepared._controller  # noqa: SLF001
    assert controller is not None

    steady_values = tuple(str(path.resolve()) for path in steady_runs)
    controller.run_filter.value = steady_values
    calls.clear()
    controller._render_comparison()  # noqa: SLF001
    steady = controller._workflow  # noqa: SLF001
    assert isinstance(steady, FakeWorkflow)
    assert calls == [(tuple(path.resolve() for path in steady_runs), "steady_flow")]
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.task == "steady_flow"
    assert prepared.selection_state.selection.channels == ("p", "u", "v")
    assert prepared.selection_state.selection.physical_time is None

    controller.run_filter.value = tuple(str(path.resolve()) for path in transient_runs)
    transient = controller._workflow  # noqa: SLF001
    assert isinstance(transient, FakeWorkflow)
    assert steady.close_count == 1
    assert prepared.selection_state.selection.task == "transient_drying"
    assert prepared.selection_state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert prepared.selection_state.selection.physical_time == 1.0
    assert tuple(task for _run_dirs, task in calls) == (
        "steady_flow",
        "transient_drying",
    )

    controller.run_filter.value = (
        str(steady_runs[0].resolve()),
        str(transient_runs[0].resolve()),
    )
    assert controller._workflow is None  # noqa: SLF001
    assert transient.close_count == 1
    assert controller.analysis.children == ()
    assert "incompatible task" in controller.status.value
    assert tuple(task for _run_dirs, task in calls) == (
        "steady_flow",
        "transient_drying",
    )
    prepared.close()


def test_comparison_rejects_same_objective_with_different_persisted_protocol(
    tmp_path: Path,
) -> None:
    """Fingerprint the complete persisted Evaluation configuration, not its objective label."""
    objective = {"id": "normalized_drying_group_macro_rmse"}
    first = _write_run(
        tmp_path,
        leaf="protocol-one",
        evaluation={"objective": objective, "metrics": [{"id": "metric"}], "rollout": {"origins": "early"}},
    )
    second = _write_run(
        tmp_path,
        leaf="protocol-two",
        evaluation={"objective": objective, "metrics": [{"id": "metric"}], "rollout": {"origins": "all"}},
    )
    by_path = {run.run_dir: run for run in discovery.discover_evaluation_runs(tmp_path).runs}
    selected = (by_path[first.resolve()], by_path[second.resolve()])

    assert {run.evaluation_protocol for run in selected} == {"normalized_drying_group_macro_rmse"}
    assert len({run.evaluation_config_identity for run in selected}) == 2
    assert workspace._compatibility_issues(selected) == (  # noqa: SLF001
        "Selected runs have incompatible evaluation protocol.",
    )


def test_comparison_metadata_rejects_incompatible_scientific_contracts(tmp_path: Path) -> None:
    """Reject scientific metadata mismatches before artifact loading."""
    run_dir = _write_run(tmp_path, leaf="reference", artifact=False)
    reference = discovery.discover_evaluation_runs(tmp_path).runs[0]
    compatible = replace(reference, run_dir=run_dir.parent / "compatible-peer")
    assert workspace._compatibility_issues((reference, compatible)) == ()  # noqa: SLF001

    mismatches = (
        (replace(compatible, task="steady_flow"), "task"),
        (replace(compatible, evaluation_protocol="different-objective"), "evaluation objective"),
        (replace(compatible, evaluation_config_identity="f" * 64), "evaluation protocol"),
        (replace(compatible, spatial_stride=8), "spatial stride"),
    )
    for peer, expected_label in mismatches:
        issues = workspace._compatibility_issues((reference, peer))  # noqa: SLF001
        assert len(issues) == 1
        assert expected_label in issues[0]


def test_evaluation_notebooks_execute_without_hardcoded_run_lists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute both thin notebook entry points against synthetic missing artifacts."""
    storage_root = tmp_path / "storage"
    experiments_root = storage_root / "03_experiments"
    for task in ("steady_flow", "transient_drying"):
        for index in range(2):
            _write_run(
                experiments_root,
                leaf=f"notebook-{task}-{index}",
                task=task,
                run_name=f"notebook-{task}-{index}",
            )
    monkeypatch.setattr(
        common.paths,
        "get_storage_root",
        lambda *, storage_root=None: Path(storage_root) if storage_root is not None else tmp_path / "storage",
    )
    import IPython.display  # noqa: PLC0415

    monkeypatch.setattr(IPython.display, "display", lambda _value: None)
    repository_root = Path(__file__).resolve().parents[2]
    for relative in (
        "notebooks/eval_single_model.ipynb",
        "notebooks/eval_comparison_models.ipynb",
    ):
        notebook = json.loads((repository_root / relative).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")
        assert "analysis.evaluation.workspace" in source
        assert "RUN_DIR" not in source
        assert "RUN_DIRS" not in source
        assert "auto_build_missing" not in source
        assert "replace_with" not in source
        namespace: dict[str, object] = {}
        exec(compile(source, relative, "exec"), namespace)  # noqa: S102
        prepared = namespace["workspace"]
        assert isinstance(prepared, workspace.PreparedEvaluationWorkspace)
        controller = prepared._controller  # noqa: SLF001
        assert controller is not None
        descriptions = set(_widget_descriptions(controller.panel))
        assert descriptions.isdisjoint({"Task:", "Stage:", "Seed:", "Regime:", "View:"})
        if prepared.kind == "single":
            assert descriptions == {"Run:"}
            options = tuple(value for _label, value in controller.run.options)
            selected_value = controller.run.value
            assert isinstance(selected_value, (str, Path))
            selected_run = controller._run_by_path(selected_value)  # noqa: SLF001
            switched = next(
                value
                for value in options
                if controller._run_by_path(value).task != selected_run.task  # noqa: SLF001
            )
            controller.run.value = switched
            assert prepared.selection_state is not None
            assert prepared.selection_state.selection.task == controller._run_by_path(switched).task  # noqa: SLF001
            assert "Artifact" in controller.status.value
        else:
            assert descriptions == {"Runs:", "Material/Dataset:"}
            assert len(controller.run_filter.options) == 4
            assert "Select at least two runs" in controller.status.value
        prepared.close()
