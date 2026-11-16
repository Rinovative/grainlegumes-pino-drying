"""Protect read-only persisted Evaluation run discovery.

Responsibilities:
  - Check persisted run catalogs, task-aware workspace dispatch, stage lineage, and comparison
    admission
  - Check the stage-history read after run admission and the study-summary read

Design principles:
  - Discovery stays read-only and distinguishes readiness from active staging
  - Histories are written by the real history writer; only the run admission is replaced by a fake lease

This module does NOT:
  - Run training or infer missing artifacts during catalog discovery
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import pytest
import yaml
from support.synthetic_task import synthetic_split_payload

from src import analysis, common
from src.analysis.artifacts import analysis_artifact_contracts as artifact_contracts
from src.analysis.evaluation import evaluation_run_discovery as discovery
from src.analysis.evaluation import evaluation_selection as selection
from src.analysis.evaluation import evaluation_workspace as workspace
from src.experiments import experiments_run_identity as identity
from src.learning.training import learning_training_history as training_history

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _metadata_config_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use compact saved metadata while retaining exact parent content checks."""
    monkeypatch.setattr(identity.config_loader, "validate_resolved_config", lambda config: config)
    monkeypatch.setattr(identity.training_checkpoint, "config_digest", identity.resolved_config_digest)


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record what each publication into a workspace panel shows: the prepared panel, or ``None`` for a cleared panel."""
    results: list[object] = []
    real = analysis.ui.viewers.render_view

    def record(view: Any, plot_func: Any, **kwargs: Any) -> bool:
        def recorded(*args: Any, **plot_kwargs: Any) -> Any:
            result = plot_func(*args, **plot_kwargs)
            results.append(result)
            return result

        return real(view, recorded, **kwargs)

    monkeypatch.setattr(analysis.ui.viewers, "render_view", record)
    return results


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
    branch_id: str | None = None,
    artifact: bool = False,
    evaluation: dict[str, Any] | None = None,
    family: str | None = None,
    trial: str | None = None,
    seed: int = 17,
    split: bool = True,
    physics: bool | None = False,
    model_kind: str | None = "fno",
    a0_run_name: str | None = None,
) -> Path:
    """Write one compact persisted leaf; its saved split holds the ID Dataset plus one additional OOD part unless ``split`` is false."""
    stage_id = comparison_arm if comparison_arm in {"a0", "b", "a1"} else "a0" if stage == "a" else "b" if stage == "b" else "single"
    layout = {
        "task": task,
        "dataset": "dataset-id",
        "family": "optuna" if trial is not None else family or leaf,
        "config_name": "fno_study" if trial is not None else "fno_m2x2_h4_l1",
        "model_label": "fno_m2x2_h4_l1",
        "seed": seed,
        "revision": None,
        "budget_epochs": 3,
        "units_per_update": 2,
        "schedule": "cosine",
        "stage": stage_id,
        "pair": f"pair-{(branch_id or 'b' * 16)[:4]}" if stage_id in {"b", "a1"} else None,
        "trial": trial,
        "a0_id": "a" * 16,
        "branch_id": (branch_id or "b" * 16) if stage_id in {"b", "a1"} else None,
    }
    run_dir = common.paths.resolve_config_run_dir({"run": {"layout": layout}, "paths": {"output_root": str(root)}})
    run_dir.mkdir(parents=True)
    name = run_name or f"scientific-{leaf}"
    config: dict[str, Any] = {
        "task": task,
        "run": {"name": name, "seed": seed, "layout": layout},
        "paths": {"output_root": str(root)},
        "data": {
            "train_dataset": "dataset-id",
            "material_ood_datasets": [],  # The OOD roles come from the saved split, never from these ids.
        },
    }
    if model_kind is not None:  # None leaves the model block out, as an incomplete saved config does.
        config["model"] = {
            "kind": model_kind,
            "params": {"n_modes": [2, 2], "hidden_channels": 4, "n_layers": 1, "positional_embedding": None, "domain_padding": 0.1},
        }
    if physics is not None:  # None leaves the loss block out, as an incomplete saved config does.
        config["loss"] = {"physics": {"enabled": physics}}
    if evaluation is not None:
        config["evaluation"] = evaluation
    if stage is not None or comparison_arm is not None or a0_run_name is not None:
        config["training"] = {}
        if stage is not None:
            config["training"]["stage"] = stage
        if comparison_arm is not None:
            config["training"]["comparison_arm"] = comparison_arm
        if a0_run_name is not None:
            config["training"]["teacher_handoff"] = {"source_run_name": a0_run_name}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "task": task,
                "run_name": name,
                "status": status,
                "updated_at": updated_at,
                "effective_config_digest": "e" * 64,
                "best_checkpoint_sha256": hashlib.sha256(b"best").hexdigest(),
                "config_sha256": common.serialization.file_sha256(run_dir / "config.yaml"),
                "checkpoint_identity": {"effective_config_digest": identity.resolved_config_digest(config)},
                "run_identity": {"resolved_config_sha256": identity.resolved_config_digest(config)},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "best_checkpoint.pt").write_bytes(b"best")
    if split:
        payload = synthetic_split_payload(task, train_dataset="dataset-id", material_ood=(("dataset-ood", ("chickpea__case_0001__00000000000a",)),))
        common.serialization.atomic_torch_save(payload, common.paths.resolve_split_indices_path(run_dir))
    if artifact:
        second_root = run_dir / "analysis" / "material_ood" / "dataset-ood"
        for artifact_root in (run_dir / "analysis" / "id", second_root):
            artifact_root.mkdir(parents=True)
            (artifact_root / "cases.parquet").write_bytes(b"parquet")
            payload_root = artifact_root / "npz"
            payload_root.mkdir()
            (payload_root / "case-0.npz").write_bytes(b"npz")
            marker = {
                "artifact_kind": discovery.transient_artifact.TRANSIENT_ARTIFACT_KIND if task == "transient_drying" else "steady_artifact",
                "task": task,
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


def _write_parent(
    root: Path,
    *,
    label: str,
    children: dict[str, Path],
) -> str:
    """Write one exact matched A0/B/A1 branch parent while leaving shared A0 unowned."""
    branch_config = yaml.safe_load((children["stage_b"] / "config.yaml").read_text())
    layout = branch_config["run"]["layout"]
    child_records: dict[str, dict[str, str]] = {}
    for role, child_path in children.items():
        config = yaml.safe_load((child_path / "config.yaml").read_text())
        child_records[role] = {
            "run_name": config["run"]["name"],
            "path": "../a0" if role == "stage_a0" else role.removeprefix("stage_"),
            "semantic_config_sha256": identity.resolved_config_digest(config),
        }
    payload = {
        "schema_kind": identity.EXPERIMENT_RECORD_SCHEMA_KIND,
        "topology": "a0_to_b_and_a1",
        "task": "transient_drying",
        "parent_label": label,
        "layout": layout,
        "seed": 17,
        "authored_config": {"path": "synthetic.yaml", "basename": "synthetic.yaml", "sha256": "a" * 64},
        "dataset_identity": identity.dataset_identity_evidence(branch_config),
        "children": child_records,
        "handoff": {
            "source_run_name": child_records["stage_a0"]["run_name"],
            "target_run_names": [child_records[role]["run_name"] for role in children if role != "stage_a0"],
            "initialization": "a0_best_full_common_fresh_controller",
        },
    }
    record = {
        **payload,
        "parent_identity_sha256": common.serialization.canonical_json_sha256(
            {key: value for key, value in payload.items() if key not in {"authored_config", "source_repository"}}
        ),
        "source_repository": {"commit": None, "dirty": None},
    }
    path = identity.experiment_record_path(record, output_root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return record["parent_identity_sha256"]


def test_dataset_binding_label_preserves_authored_readable_name() -> None:
    """Do not parse or strip a readable Dataset name into guessed scientific identity."""
    id_dataset = "lentil_chickpea_four_hour"
    additional_dataset = "kidney_bean_four_hour"

    assert (
        workspace._EvaluationWorkspaceController._dataset_binding_label(
            dataset_name=id_dataset,
            role="id",
        )
        == "lentil_chickpea_four_hour · ID"
    )
    assert (
        workspace._EvaluationWorkspaceController._dataset_binding_label(
            dataset_name=additional_dataset,
            role="material_ood",
            declared_role="material_ood",
        )
        == "kidney_bean_four_hour · Material OOD"
    )


def test_discovers_branch_and_shared_a0_with_exact_identity(tmp_path: Path) -> None:
    """Admit shared A0 independently and bind branch children to their exact parent."""
    stage_a = _write_run(tmp_path, leaf="storage-a", family="shared", run_name="scientific-a", comparison_arm="a0")
    stage_b = _write_run(tmp_path, leaf="storage-b", family="shared", run_name="scientific-b", comparison_arm="b")
    stage_a1 = _write_run(tmp_path, leaf="storage-a1", family="shared", run_name="scientific-a1", comparison_arm="a1")
    legacy = _write_run(tmp_path, leaf="opaque-legacy", task="steady_flow", run_name="exact-steady")
    parent_identity = _write_parent(
        tmp_path,
        label="concise-parent",
        children={"stage_a0": stage_a, "stage_b": stage_b, "stage_a1": stage_a1},
    )

    catalog = discovery.discover_evaluation_runs(tmp_path)

    assert len(catalog.runs) == len({"scientific-a", "scientific-b", "scientific-a1", "exact-steady"})
    assert len(catalog.groups) == len({"branch-parent", "shared-a0", "independent-run"})
    grouped = next(group for group in catalog.groups if group.identity_sha256 == parent_identity)
    assert {item.run_name for item in grouped.children} == {"scientific-b", "scientific-a1"}
    legacy_record = next(item for item in catalog.runs if item.run_name == "exact-steady")
    assert legacy_record.run_dir == legacy.resolve()
    assert legacy_record.parent_identity_sha256 is None
    assert legacy_record.dataset_id == "dataset-id"
    assert legacy_record.identity_format == "independent"
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
    assert available_record.artifact_command == (
        "./scripts/slurm_ml.sh --mode cpu --partition standard --cpus-per-task 4 --mem 16G --time 02:00:00 "
        f"artifacts --run-dir {available.resolve()} --split id --split material_ood"
    )  # full coverage names every planned held-out role


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
    assert record.id_artifact is not None
    assert record.id_artifact.state == "ready"
    assert record.ood_artifacts["material_ood"] is not None
    assert record.ood_artifacts["material_ood"].state == "ready"
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
    stage = target.parent / ".id.stage"
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
    failed_stage = run_dir / "analysis" / ".id.stage"
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
    controller = workspace._EvaluationWorkspaceController.__new__(workspace._EvaluationWorkspaceController)
    controller._labels = {record.run_dir: "FNO · seed 17"}
    html = controller._status_html(
        record,
        selected_roles=("id", "material_ood"),
        material_coverage=(
            ("lentil", "id", "id"),
            ("chickpea", "id", "id"),
            ("kidney_bean", "ood", "material_ood"),
        ),
    )

    assert "Lentil · ID · available" in html
    assert "Chickpea · ID · available" in html
    assert "Kidney bean · OOD · available" in html
    assert "Drying group macro RMSE" in html
    assert re.search(r"Spatial stride</th><td[^>]*>1</td>", html)
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
    controller = prepared._controller
    assert controller is not None
    assert calls[-1]["artifact_roots"] == {"id": scoped}
    assert calls[-1]["sections"] == ("sample_viewer",)
    assert "ID · scoped partial (1 case)" in controller._details
    assert "Generate full artifact coverage" in controller._details
    assert "--one-case" not in controller._details
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

    def counted(root: Path | str, **kwargs: Any) -> discovery.EvaluationRunCatalog:
        scans.append(Path(root))
        return original(root, **kwargs)

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
    controller = prepared._controller
    assert controller is not None
    assert controller.runs.header.value is True
    controller.runs.select(controller.runs.selected())
    assert controller.runs.header.value is True  # only a user pick closes the list, not select()
    grid = controller.view.widget
    assert len(grid.children) == 3
    assert grid.children[:2] == (controller.view.content, controller.runs.body)
    assert controller.panel.children[1] is grid
    assert isinstance(controller.panel.children[0], widgets.HTML)
    assert len(scans) == 1
    assert controller.runs.boxes
    assert "ID · missing; Material OOD · missing" in controller._details
    assert "slurm_ml.sh" in controller._details
    assert "artifacts --run-dir" in controller._details
    assert "--one-case" not in controller._details
    assert controller.view.status.level == "warning"
    prepared.close()

    compared = workspace.prepare_model_comparison_evaluation_workspace(experiments_root=experiments_root)
    assert len(scans) == 2
    comparison_controller = compared._controller
    assert comparison_controller is not None
    comparison_controller.runs.select(comparison_controller.runs.selected())
    assert comparison_controller.runs.header.value is True  # a multiple choice stays open until its header is clicked
    assert "Select at least two runs" in comparison_controller.view.status.text
    assert "Select at least two runs" in comparison_controller._details
    compared.close()


def test_comparison_workspace_admits_exact_a0_a1_b_stage_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let the workflow validate one exact three-arm transient comparison."""
    experiments_root = tmp_path / "03_experiments"
    experiment = "three-arm"
    a0 = _write_run(
        experiments_root,
        leaf="a0",
        family=experiment,
        run_name="scientific-a0",
        comparison_arm="a0",
        artifact=True,
    )
    branch_id = "c" * 16
    branch_runs = {
        arm: _write_run(
            experiments_root,
            leaf=arm,
            family=experiment,
            run_name=f"scientific-{arm}",
            comparison_arm=arm,
            branch_id=branch_id,
            artifact=True,
        )
        for arm in ("a1", "b")
    }
    run_dirs = (a0, branch_runs["a1"], branch_runs["b"])
    _write_parent(
        experiments_root,
        label="three-arm-parent",
        children={"stage_a0": a0, "stage_b": branch_runs["b"], "stage_a1": branch_runs["a1"]},
    )
    calls: list[tuple[Path, ...]] = []

    class FakeWorkflow:
        def __init__(self) -> None:
            self.panel = widgets.Label("three-arm")
            self.refused_roles: dict[str, str] = {}
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def prepare(selected: tuple[Path, ...], **_kwargs: object) -> FakeWorkflow:
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
    controller = prepared._controller
    assert controller is not None

    controller.runs.select([str(path.resolve()) for path in run_dirs])

    assert calls[-1] == tuple(path.resolve() for path in run_dirs)
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.run_dirs == calls[-1]
    assert "incompatible stage" not in controller._details
    assert controller.view.status.level == "normal"
    prepared.close()


def test_matched_a1_discovery_keeps_distinct_stage_and_parent_group(tmp_path: Path) -> None:
    """Expose one A0/B/A1 parent with its distinct stages and parent group."""
    experiments_root = tmp_path / "03_experiments"
    branch_id = "d" * 16
    run_dirs = {
        arm: _write_run(
            experiments_root,
            leaf=arm,
            family="matched",
            run_name=f"scientific-{arm}",
            comparison_arm=arm,
            branch_id=branch_id,
            artifact=True,
            a0_run_name=None if arm == "a0" else "scientific-a0",
        )
        for arm in ("a0", "b", "a1")
    }
    parent_identity = _write_parent(
        experiments_root,
        label="matched-parent",
        children={role: run_dirs[role.removeprefix("stage_")] for role in ("stage_a0", "stage_b", "stage_a1")},
    )

    catalog = discovery.discover_evaluation_runs(experiments_root)
    assert catalog.issues == ()
    by_name = {run.run_name: run for run in catalog.runs}
    assert {name: run.stage for name, run in by_name.items()} == {
        "scientific-a0": "A0",
        "scientific-b": "B",
        "scientific-a1": "A1",
    }
    assert {run.parent_identity_sha256 for run in catalog.runs} == {None, parent_identity}
    assert {name: run.a0_run_name for name, run in by_name.items()} == {
        "scientific-a0": None,
        "scientific-b": "scientific-a0",
        "scientific-a1": "scientific-a0",
    }
    assert workspace._compatibility_issues(catalog.runs, _labels_of(catalog.runs)) == ()
    relabeled = tuple(replace(run, stage="A0") if run.stage == "B" else run for run in catalog.runs)
    # Stages are selectable attributes; controlled lineage has a separate gate.
    assert workspace._compatibility_issues(relabeled, _labels_of(relabeled)) == ()


def test_two_matched_branches_share_one_independent_a0_without_parent_conflicts(tmp_path: Path) -> None:
    """Keep two matched A1/B budgets as distinct branches from the exact shared A0."""
    experiments_root = tmp_path / "03_experiments"
    experiment = "shared-budget-comparison"
    stage_a0 = _write_run(
        experiments_root,
        leaf="a0",
        family=experiment,
        run_name="shared-a0",
        comparison_arm="a0",
    )
    branches = {
        token: {
            arm: _write_run(
                experiments_root,
                leaf=f"{token}-{arm}",
                family=experiment,
                run_name=f"{token}-branch-{arm}",
                comparison_arm=arm,
                branch_id=branch_id,
            )
            for arm in ("b", "a1")
        }
        for token, branch_id in (("first", "1" * 16), ("second", "2" * 16))
    }
    parents = {
        token: _write_parent(
            experiments_root,
            label=f"{token}-branch-parent",
            children={"stage_a0": stage_a0, "stage_b": runs["b"], "stage_a1": runs["a1"]},
        )
        for token, runs in branches.items()
    }

    catalog = discovery.discover_evaluation_runs(experiments_root)

    assert catalog.issues == ()
    by_name = {run.run_name: run for run in catalog.runs}
    assert by_name["shared-a0"].stage == "A0"
    assert by_name["shared-a0"].parent_identity_sha256 is None
    for token, parent in parents.items():
        assert by_name[f"{token}-branch-b"].parent_identity_sha256 == parent
        assert by_name[f"{token}-branch-a1"].stage == "A1"
        assert by_name[f"{token}-branch-a1"].parent_identity_sha256 == parent
    assert parents["first"] != parents["second"]


def test_single_workspace_switches_transient_steady_transient_without_state_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: list[object],
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
    controller = prepared._controller
    assert controller is not None

    controller.runs.select([str(transient_run.resolve())])
    calls.clear()
    controller._render_single()
    transient_first = controller._workflow
    assert isinstance(transient_first, FakeWorkflow)
    assert calls == [(transient_run.resolve(), "transient_drying")]
    assert published[-1] is transient_first.panel
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.task == "transient_drying"
    assert prepared.selection_state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert prepared.selection_state.selection.physical_time == 1.0
    assert prepared.selection_state.selection.protocol == "autonomous_full"
    assert prepared.selection_state.selection.horizon == "full"

    controller.runs.select([str(steady_run.resolve())])
    steady = controller._workflow
    assert isinstance(steady, FakeWorkflow)
    assert transient_first.close_count == 1
    assert published[-1] is steady.panel
    assert prepared.selection_state.selection.task == "steady_flow"
    assert prepared.selection_state.selection.channels == ("p", "u", "v")
    assert prepared.selection_state.selection.physical_time is None
    assert prepared.selection_state.selection.protocol is None
    assert prepared.selection_state.selection.horizon is None

    controller.runs.select([str(transient_run.resolve())])
    transient_second = controller._workflow
    assert isinstance(transient_second, FakeWorkflow)
    assert transient_second is not transient_first
    assert steady.close_count == 1
    assert published[-1] is transient_second.panel
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


@pytest.mark.parametrize("older_load_fails", [False, True], ids=["older-load-finishes", "older-load-fails"])
def test_a_selection_made_while_another_loads_is_the_one_that_stays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, older_load_fails: bool) -> None:
    """The older load can neither replace the newer panel and its summary nor, by failing, clear them."""
    experiments_root = tmp_path / "03_experiments"
    transient_run = _write_run(experiments_root, leaf="older", task="transient_drying", run_name="older-run", artifact=True)
    steady_run = _write_run(experiments_root, leaf="newer", task="steady_flow", run_name="newer-run", artifact=True)
    workflows: dict[str, Any] = {}
    interrupt: list[bool] = []

    class FakeWorkflow:
        def __init__(self, task: str) -> None:
            self.task = task
            self.panel = widgets.Label(f"{task}-panel")
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    def prepare(_run_dir: Path, **kwargs: Any) -> FakeWorkflow:
        task = kwargs["expected_task"]
        kwargs["selection_state"].bind_capabilities(
            selection.EvaluationViewCapabilities(task="steady_flow", channels=("p",), case_ids=("case",))
            if task == "steady_flow"
            else selection.EvaluationViewCapabilities(task="transient_drying", channels=("T",), case_ids=("case",), physical_times=(0.0,))
        )
        loaded = workflows.setdefault(task, FakeWorkflow(task))
        if interrupt and task == "transient_drying":
            interrupt.clear()
            assert controller is not None
            controller.runs.select([str(steady_run.resolve())])  # the user chooses another run while this one loads
            if older_load_fails:
                message = "the older load failed"
                raise RuntimeError(message)
        return loaded

    monkeypatch.setattr(workspace.workflow, "prepare_single_model_evaluation_workspace", prepare)
    prepared = workspace.prepare_single_model_evaluation_workspace(experiments_root=experiments_root)
    controller = prepared._controller
    assert controller is not None
    workflows.clear()

    interrupt.append(True)
    controller.runs.select([str(transient_run.resolve())])

    newer, older = workflows["steady_flow"], workflows["transient_drying"]
    assert controller._workflow is newer
    assert controller._labels[steady_run.resolve()] in controller.view.status.text
    assert controller.view.status.level == "normal"
    assert "older load failed" not in controller._details
    assert older.close_count == (0 if older_load_fails else 1)
    assert newer.close_count == 0
    prepared.close()


def test_a_workflow_prepared_for_a_single_run_load_that_then_fails_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An inventory that raises after preparation closes the prepared session and the panel shows the failure."""
    experiments_root = tmp_path / "03_experiments"
    _write_run(experiments_root, leaf="late-failure", task="steady_flow", artifact=True)
    prepared_workflows: list[Any] = []

    class FakeSession:
        def partitioned_case_inventory(self) -> tuple[()]:
            message = "inventory failed"
            raise RuntimeError(message)

    class FakeWorkflow:
        def __init__(self) -> None:
            self.panel = widgets.Label("late-failure")
            self.session = FakeSession()
            self.close_count = 0

        def close(self) -> None:
            self.close_count += 1

    def prepare(_run_dir: Path, **_kwargs: Any) -> FakeWorkflow:
        prepared_workflows.append(FakeWorkflow())
        return prepared_workflows[-1]

    monkeypatch.setattr(workspace.workflow, "prepare_single_model_evaluation_workspace", prepare)
    prepared = workspace.prepare_single_model_evaluation_workspace(experiments_root=experiments_root)
    controller = prepared._controller
    assert controller is not None

    (workflow,) = prepared_workflows
    assert workflow.close_count == 1
    assert controller._workflow is None
    assert controller.view.failure.shown
    assert controller.view.status.level == "error"
    prepared.close()


def test_a_workflow_prepared_for_a_comparison_load_that_then_fails_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reading the refused roles of a prepared comparison may fail; the prepared session is closed all the same."""
    for name in ("first-experiment", "second-experiment"):
        _write_run(tmp_path, leaf=name, family=name, artifact=True)
    prepared_workflows: list[Any] = []

    class FakeWorkflow:
        def __init__(self) -> None:
            self.panel = widgets.Label("comparison")
            self.close_count = 0

        @property
        def refused_roles(self) -> dict[str, str]:
            message = "refused roles unreadable"
            raise RuntimeError(message)

        def close(self) -> None:
            self.close_count += 1

    def prepare(_run_dirs: tuple[Path, ...], **_kwargs: Any) -> FakeWorkflow:
        prepared_workflows.append(FakeWorkflow())
        return prepared_workflows[-1]

    monkeypatch.setattr(workspace.workflow, "prepare_model_comparison_evaluation_workspace", prepare)
    prepared = workspace.prepare_model_comparison_evaluation_workspace(experiments_root=tmp_path)
    controller = prepared._controller
    assert controller is not None

    (workflow,) = prepared_workflows
    assert workflow.close_count == 1
    assert controller._workflow is None
    assert controller.view.failure.shown
    prepared.close()


def test_comparison_workspace_switches_tasks_and_rejects_mixed_task_hybrids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    published: list[object],
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
            self.refused_roles: dict[str, str] = {}
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
    controller = prepared._controller
    assert controller is not None

    steady_values = [str(path.resolve()) for path in steady_runs]
    controller.runs.select(steady_values)
    calls.clear()
    controller._render_comparison()
    steady = controller._workflow
    assert isinstance(steady, FakeWorkflow)
    assert published[-1] is steady.panel
    assert calls == [(tuple(path.resolve() for path in steady_runs), "steady_flow")]
    assert prepared.selection_state is not None
    assert prepared.selection_state.selection.task == "steady_flow"
    assert prepared.selection_state.selection.channels == ("p", "u", "v")
    assert prepared.selection_state.selection.physical_time is None

    controller.runs.select([str(path.resolve()) for path in transient_runs])
    transient = controller._workflow
    assert isinstance(transient, FakeWorkflow)
    assert published[-1] is transient.panel
    assert steady.close_count == 1
    assert prepared.selection_state.selection.task == "transient_drying"
    assert prepared.selection_state.selection.channels == ("T", "phi", "w_surf", "w_int")
    assert prepared.selection_state.selection.physical_time == 1.0
    assert tuple(task for _run_dirs, task in calls) == (
        "steady_flow",
        "transient_drying",
    )

    controller.runs.select([str(steady_runs[0].resolve()), str(transient_runs[0].resolve())])
    assert controller._workflow is None
    assert transient.close_count == 1
    assert published[-1] is None
    assert controller.view.status.level == "warning"
    assert "incompatible task" in controller._details
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
    assert workspace._compatibility_issues(selected, _labels_of(selected)) == ("Selected runs have incompatible evaluation protocol.",)


def test_comparison_metadata_rejects_incompatible_scientific_contracts(tmp_path: Path) -> None:
    """Reject scientific metadata mismatches before artifact loading."""
    run_dir = _write_run(tmp_path, leaf="reference", artifact=False)
    reference = discovery.discover_evaluation_runs(tmp_path).runs[0]
    compatible = replace(reference, run_dir=run_dir.parent / "compatible-peer")
    assert workspace._compatibility_issues((reference, compatible), _labels_of((reference, compatible))) == ()

    mismatches = (
        (replace(compatible, task="steady_flow"), "task"),
        (replace(compatible, evaluation_protocol="different-objective"), "evaluation objective"),
        (replace(compatible, evaluation_config_identity="f" * 64), "evaluation protocol"),
        (replace(compatible, spatial_stride=8), "spatial stride"),
    )
    for peer, expected_label in mismatches:
        issues = workspace._compatibility_issues((reference, peer), _labels_of((reference, peer)))
        assert len(issues) == 1
        assert expected_label in issues[0]


def _widget_tree(widget: widgets.Widget) -> tuple[widgets.Widget, ...]:
    """Return one widget tree in depth-first order."""
    return (widget, *(item for child in getattr(widget, "children", ()) for item in _widget_tree(child)))


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
        "notebooks/05_eval_single_model.ipynb",
        "notebooks/06_eval_comparison_models.ipynb",
    ):
        notebook = json.loads((repository_root / relative).read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"] if cell["cell_type"] == "code")
        namespace: dict[str, object] = {}
        exec(compile(source, relative, "exec"), namespace)  # noqa: S102
        prepared = namespace["workspace"]
        assert isinstance(prepared, workspace.PreparedEvaluationWorkspace)
        # No output widget in the shell: VS Code's Jupyter extension stalls on output-capture updates (item 177).
        assert prepared.panel is not None
        assert not any(isinstance(item, widgets.Output) for item in _widget_tree(prepared.panel))
        controller = prepared._controller
        assert controller is not None
        if prepared.kind == "single":
            options = tuple(str(key) for key in controller.runs.boxes)
            (selected_value,) = controller.runs.selected()
            assert isinstance(selected_value, (str, Path))
            selected_run = controller._run_by_path(selected_value)
            switched = next(value for value in options if controller._run_by_path(value).task != selected_run.task)
            controller.runs.select([switched])
            assert prepared.selection_state is not None
            assert prepared.selection_state.selection.task == controller._run_by_path(switched).task
        else:
            assert len(controller.runs.boxes) == 4
        prepared.close()


def test_optuna_trials_are_skipped_by_recorded_layout_without_hashing_their_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip trial leaves (with or without a kept checkpoint) while a normal run is still listed."""
    experiments_root = tmp_path / "03_experiments"
    kept = _write_run(experiments_root, leaf="trial-kept", trial="0007__fno_m2x2_h4_l1__lr-0p001")
    removed = _write_run(experiments_root, leaf="trial-removed", trial="0008__fno_m2x2_h4_l1__lr-0p002")
    (removed / "best_checkpoint.pt").unlink()
    normal = _write_run(experiments_root, leaf="normal")
    study_root = common.paths.resolve_study_dir("transient_drying", "dataset-id", "fno_study", 3, output_root=experiments_root)
    assert kept.parent == removed.parent == study_root / "trials"
    _forbid_checkpoint_hashing(monkeypatch)

    from_experiments = discovery.discover_evaluation_runs(experiments_root)
    from_study = discovery.discover_evaluation_runs(study_root)

    assert [run.run_dir for run in from_experiments.runs] == [normal.resolve()]
    assert from_study.runs == ()
    assert from_experiments.issues == from_study.issues == ()


def test_normal_run_without_best_checkpoint_stays_listed_but_not_evaluable(tmp_path: Path) -> None:
    """Fail closed for a regular run whose best checkpoint is missing."""
    run_dir = _write_run(tmp_path, leaf="no-checkpoint", artifact=True)
    (run_dir / "best_checkpoint.pt").unlink()

    record = discovery.discover_evaluation_runs(tmp_path).runs[0]

    assert record.run_dir == run_dir.resolve()
    assert record.checkpoints.best_digest is None
    assert not record.evaluable
    assert not record.action_enabled


def _forbid_checkpoint_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any file hashing during discovery fail the test."""

    def refuse(value: Path | str) -> str:
        msg = f"discovery must not hash {value}"
        raise AssertionError(msg)

    monkeypatch.setattr(common.serialization, "file_sha256", refuse)
    monkeypatch.setattr(common.serialization, "cached_file_sha256", refuse)


def test_discovery_reads_no_checkpoint_bytes_and_exposes_the_recorded_best_digest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the best digest from the summary and leave byte verification to selection."""
    _write_run(tmp_path, leaf="recorded")
    _forbid_checkpoint_hashing(monkeypatch)

    record = discovery.discover_evaluation_runs(tmp_path).runs[0]

    assert record.checkpoints.best_digest == hashlib.sha256(b"best").hexdigest()
    assert record.evaluable


@pytest.mark.parametrize("recorded", [None, "", "A" * 64, "a" * 63, 7])
def test_missing_or_malformed_recorded_best_digest_makes_the_run_not_evaluable(tmp_path: Path, recorded: object) -> None:
    """Fail closed when the summary carries no lowercase 64-hex best digest."""
    run_dir = _write_run(tmp_path, leaf="unrecorded")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["best_checkpoint_sha256"] = recorded
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    record = discovery.discover_evaluation_runs(tmp_path).runs[0]

    assert record.checkpoints.best_digest is None
    assert not record.evaluable
    assert not record.action_enabled


def test_comparison_labels_carry_seed(tmp_path: Path) -> None:
    """Label runs of one experiment by seed."""
    _write_run(tmp_path, leaf="first", family="shared", seed=17)
    _write_run(tmp_path, leaf="second", family="shared", seed=18)
    runs = discovery.discover_evaluation_runs(tmp_path).runs
    shared_dir = tmp_path / "transient_drying" / "dataset-id" / "shared"
    shared = tuple(run for run in runs if shared_dir in run.run_dir.parents)
    assert len(shared) == 2
    assert {run.seed for run in shared} == {17, 18}
    labels = workspace._run_labels(shared)
    assert all("seed " in label for label in labels)
    assert len(set(labels)) == 2


def test_comparison_labels_name_the_model_family_by_its_display_name(tmp_path: Path) -> None:
    """A plain and a physics-informed run of one architecture and seed read FNO and PI-FNO, not their architecture descriptor."""
    _write_run(tmp_path, leaf="plain", family="plain", seed=9)
    _write_run(tmp_path, leaf="physics", family="physics", seed=9, physics=True)
    runs = sorted(discovery.discover_evaluation_runs(tmp_path).runs, key=lambda run: run.model_family or "")
    assert [run.model_family for run in runs] == ["fno", "pi-fno"]
    assert workspace._run_labels(tuple(runs)) == ("FNO · seed 9", "PI-FNO · seed 9")


def _labels_of(runs: Any) -> dict[Path, str]:
    """Return the short label of each run by its directory, as the workspace controller holds them."""
    return dict(zip((run.run_dir for run in runs), workspace._run_labels(runs), strict=True))


def _label_run(
    stage: str | None,
    *,
    family: str | None = "fno",
    seed: int | None = 9,
    run_dir: str = "/x/a",
    pair: str | None = None,
    run_name: str | None = None,
) -> Any:
    """Return the fields of a discovered run that its short label reads."""
    return SimpleNamespace(
        model_family=family,
        model_label="fno_m2x2_h4_l1",
        stage=stage,
        seed=seed,
        pair_label=pair,
        config_label=None,
        dataset_id=None,
        run_name=run_name or Path(run_dir).name,
        run_dir=Path(run_dir),
    )


def test_run_label_names_architecture_stage_and_seed_and_omits_an_absent_stage() -> None:
    """The label reads architecture, stage, seed; a steady run has no stage part."""
    runs = (_label_run("B"), _label_run("A1", run_dir="/x/b"), _label_run(None, family="pi-uno", run_dir="/x/c"))
    assert workspace._run_labels(runs) == ("FNO · B · seed 9", "FNO · A1 · seed 9", "PI-U-NO · seed 9")


def test_run_labels_stay_unique_through_the_shortest_separating_facet() -> None:
    """Two runs of one architecture, stage and seed get the facet that tells them apart."""
    runs = (_label_run("B", run_dir="/x/ep-100/b", pair="ep-100"), _label_run("B", run_dir="/x/ep-200/b", pair="ep-200"))
    labels = workspace._run_labels(runs)
    assert len(set(labels)) == 2
    assert all(label.startswith("FNO · B · seed 9 · ") for label in labels)


def test_run_labels_stay_unique_when_no_facet_separates_the_runs() -> None:
    """Runs no facet can tell apart still get unique labels (full run name, then directory) instead of failing the workspace."""
    runs = (_label_run("B", run_dir="/x/a/b", run_name="same"), _label_run("B", run_dir="/x/b/a", run_name="same"))
    labels = workspace._run_labels(runs)
    assert len(set(labels)) == 2
    assert all(label.startswith("FNO · B · seed 9 · same") for label in labels)


def test_run_label_without_a_resolved_family_keeps_the_model_descriptor() -> None:
    """A missing or unregistered family falls back to the architecture descriptor instead of failing."""
    labels = workspace._run_labels((_label_run(None, family=None), _label_run(None, family="bogus", run_dir="/x/b", seed=None)))
    assert labels == ("fno_m2x2_h4_l1 · seed 9", "fno_m2x2_h4_l1 · seed unavailable")


def test_details_table_labels_are_bold_nowrap_and_values_left_aligned(tmp_path: Path) -> None:
    """Labels stand on one line at the top of their row, values left-aligned right behind them, the model row names label and descriptor."""
    _write_run(tmp_path, leaf="table")
    record = discovery.discover_evaluation_runs(tmp_path).runs[0]
    controller = workspace._EvaluationWorkspaceController.__new__(workspace._EvaluationWorkspaceController)
    controller._labels = {record.run_dir: "FNO · seed 17"}
    html = controller._status_html(record, selected_roles=())
    assert html.count("<th style='font-weight:bold;text-align:left;vertical-align:top;white-space:nowrap;") == html.count("<th ") == 12
    assert html.count("<td style='text-align:left;vertical-align:top'>") == 12
    assert "margin-left:0" in html.split("<table style='", 1)[1].split("'", 1)[0]
    assert f"Model</th><td style='text-align:left;vertical-align:top'>FNO · seed 17 ({record.model_label})</td>" in html
    controller._labels = {record.run_dir: f"{record.model_label} · seed 17"}  # a descriptor fallback label already names the descriptor
    assert f"'>{record.model_label} · seed 17</td>" in controller._status_html(record, selected_roles=())


def test_ood_role_follows_the_saved_split_and_an_absent_split_has_none(tmp_path: Path) -> None:
    """Name the material-OOD artifact folder from the saved split, not config ids, and give a run without a split no role and no issue."""
    with_split = _write_run(tmp_path, leaf="with-split", artifact=True)
    without_split = _write_run(tmp_path, leaf="without-split", split=False)

    catalog = discovery.discover_evaluation_runs(tmp_path)

    by_dir = {run.run_dir: run for run in catalog.runs}
    split_run = by_dir[with_split.resolve()]
    assert split_run.ood_dataset_ids == {"material_ood": ("dataset-ood",)}
    assert [item["dataset_id"] for item in split_run.evaluation_assignments["material_ood"]] == ["dataset-ood"]
    assert split_run.ood_artifacts["material_ood"] is not None
    assert split_run.ood_artifacts["material_ood"].root == with_split.resolve() / "analysis" / "material_ood" / "dataset-ood"
    assert split_run.ood_artifacts["material_ood"].state == "ready"
    bare_run = by_dir[without_split.resolve()]
    assert bare_run.ood_dataset_ids == {}
    assert bare_run.evaluation_assignments == {}
    assert bare_run.ood_artifacts == {}
    assert catalog.issues == ()


def test_unreadable_saved_split_remains_visible_and_disables_actions(tmp_path: Path) -> None:
    """Show a saved run without claiming unknown Dataset roles or admitting outputs."""
    run_dir = _write_run(tmp_path, leaf="unreadable-split", artifact=True)
    common.paths.resolve_split_indices_path(run_dir).write_bytes(b"not a saved split")

    catalog = discovery.discover_evaluation_runs(tmp_path)

    assert len(catalog.runs) == 1
    run = catalog.runs[0]
    assert run.run_dir == run_dir
    assert run.metadata_issues
    assert run.metadata_issues[0].startswith("Saved split unavailable: ValueError")
    assert run.ood_artifacts == {}
    assert not run.action_enabled  # The ID artifact is ready; only the unreadable split blocks the action.
    assert any("Saved split unavailable" in issue.reason for issue in catalog.issues)


def test_model_family_comes_from_the_saved_config_and_a_missing_physics_block_keeps_the_run_listed(tmp_path: Path) -> None:
    """Resolve the plain and physics-informed family; an unresolvable one stays visible as None with an issue."""
    plain = _write_run(tmp_path, leaf="plain", artifact=True, physics=False)
    informed = _write_run(tmp_path, leaf="informed", artifact=True, physics=True)
    incomplete = _write_run(tmp_path, leaf="incomplete", artifact=True, physics=None)

    catalog = discovery.discover_evaluation_runs(tmp_path)

    by_dir = {run.run_dir: run for run in catalog.runs}
    assert (by_dir[plain.resolve()].model_family, by_dir[informed.resolve()].model_family) == ("fno", "pi-fno")
    assert by_dir[plain.resolve()].action_enabled
    unresolved = by_dir[incomplete.resolve()]
    assert unresolved.model_family is None
    assert unresolved.model_kind == "fno"
    assert unresolved.metadata_issues[0].startswith("Model family unavailable: KeyError")
    assert not unresolved.action_enabled
    assert [issue.path for issue in catalog.issues] == [unresolved.run_dir]


def test_a_missing_or_unregistered_model_block_keeps_the_run_listed_disabled_with_a_fallback_label(tmp_path: Path) -> None:
    """Resolve neither family nor label from such a block, yet list the run with a metadata issue and its action disabled."""
    missing = _write_run(tmp_path, leaf="no-model", artifact=True, model_kind=None)
    unregistered = _write_run(tmp_path, leaf="odd-model", artifact=True, model_kind="bogus")

    catalog = discovery.discover_evaluation_runs(tmp_path)

    by_dir = {run.run_dir: run for run in catalog.runs}
    for path, name in ((missing, "scientific-no-model"), (unregistered, "scientific-odd-model")):
        run = by_dir[path.resolve()]
        assert run.model_label == run.label == name
        assert any(issue.startswith("Model label unavailable:") for issue in run.metadata_issues)
        assert not run.action_enabled
    assert by_dir[missing.resolve()].model_family is None
    assert by_dir[unregistered.resolve()].model_family == "bogus"


def test_colliding_model_labels_are_distinct_at_actual_controller_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Open compatible cross-experiment models with distinct readable presentation labels."""
    paths = tuple(_write_run(tmp_path, leaf=name, family=name, artifact=True) for name in ("first-experiment", "second-experiment"))
    calls: list[tuple[tuple[Path, ...], tuple[str, ...]]] = []

    def prepare(run_dirs: tuple[Path, ...], **kwargs: Any) -> Any:
        labels = kwargs["labels"]
        assert len(set(labels)) == len(labels) == 2
        calls.append((run_dirs, labels))
        return type("SavedComparison", (), {"panel": widgets.Label("saved comparison"), "refused_roles": {}, "close": lambda _self: None})()

    monkeypatch.setattr(workspace.workflow, "prepare_model_comparison_evaluation_workspace", prepare)
    prepared = workspace.prepare_model_comparison_evaluation_workspace(experiments_root=tmp_path)
    assert calls
    selected, labels = calls[-1]
    assert set(selected) == set(paths)
    for path, label in zip(selected, labels, strict=True):
        assert path.relative_to(tmp_path).parts[2].split("-")[0] in label
        assert label.startswith("FNO · seed 17")
    controller = prepared._controller
    assert controller is not None
    listed_names = tuple(str(box.description) for box in controller.runs.boxes.values())
    assert len(set(listed_names)) == len(listed_names) == 2
    prepared.close()


def test_single_workspace_passes_the_short_label_and_shows_it_in_the_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A single run reaches the workflow under its short label, not the storage alias, and the selector header names it."""
    _write_run(tmp_path, leaf="only", artifact=True)
    labels: list[str] = []

    def prepare(_run_dir: Path, **kwargs: Any) -> Any:
        labels.append(kwargs["label"])
        return type("SavedSingle", (), {"panel": widgets.Label("saved single"), "close": lambda _self: None})()

    monkeypatch.setattr(workspace.workflow, "prepare_single_model_evaluation_workspace", prepare)
    prepared = workspace.prepare_single_model_evaluation_workspace(experiments_root=tmp_path)
    assert labels == ["FNO · seed 17"]
    controller = prepared._controller
    assert controller is not None
    assert controller.runs.header.summary == "Lauf: FNO · seed 17"
    prepared.close()


def test_labels_come_from_the_whole_catalog_so_an_unselected_run_still_separates_the_selected_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs of one architecture and seed carry their distinguishing facet in the header and the legend label, whichever is selected."""
    for name in ("first-experiment", "second-experiment"):
        _write_run(tmp_path, leaf=name, family=name, artifact=True)
    labels: list[str] = []

    def prepare(_run_dir: Path, **kwargs: Any) -> Any:
        labels.append(kwargs["label"])
        return type("SavedSingle", (), {"panel": widgets.Label("saved single"), "close": lambda _self: None})()

    monkeypatch.setattr(workspace.workflow, "prepare_single_model_evaluation_workspace", prepare)
    prepared = workspace.prepare_single_model_evaluation_workspace(experiments_root=tmp_path)
    controller = prepared._controller
    assert controller is not None
    assert len(set(controller._labels.values())) == 2
    first = labels[-1]
    assert first.startswith("FNO · seed 17 · ")
    assert controller.runs.header.summary == f"Lauf: {first}"
    other = next(key for key in controller.runs.boxes if key not in controller.runs.selected())
    controller.runs.select([other])
    assert labels[-1] != first
    assert labels[-1].startswith("FNO · seed 17 · ")
    assert controller.runs.header.summary == f"Lauf: {labels[-1]}"
    prepared.close()


def test_a_b_or_a1_run_stands_below_its_listed_a0_and_an_orphan_stands_unindented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The selector places a run under the A0 its saved teacher handoff names; a run whose A0 is not listed keeps its name-order place."""
    for leaf, name, a0 in (("a", "u__a0", None), ("b", "u__b-a1__b", "u__a0"), ("c", "z__b-a1__a1", "missing__a0")):
        _write_run(tmp_path, leaf=leaf, run_name=name, a0_run_name=a0)
    catalog = discovery.discover_evaluation_runs(tmp_path)
    assert catalog.issues == ()
    runs = tuple(sorted(catalog.runs, key=lambda run: run.run_name))
    assert [run.a0_run_name for run in runs] == [None, "u__a0", "missing__a0"]
    monkeypatch.setattr(workspace._EvaluationWorkspaceController, "_render", lambda _self: None)
    controller = workspace._EvaluationWorkspaceController(replace(catalog, runs=runs), kind="single", title="t", evaluation_compute_device="cpu")
    assert [str(box.description) for box in controller.runs.boxes.values()] == ["u__a0", "u__b-a1__b", "z__b-a1__a1"]
    rows = [row for row in cast("widgets.Box", controller.runs.body).children if isinstance(row, widgets.HBox)]
    margins = [next(row.layout.margin for row in rows if row.children == (box,)) for box in controller.runs.boxes.values()]
    assert margins == [None, "0 0 0 24px", None]


def test_relocated_current_single_bundle_stays_discoverable(tmp_path: Path) -> None:
    """Keep standalone current metadata navigable after the bundle left its authored location."""
    original = _write_run(tmp_path, leaf="authored-experiment")
    moved = tmp_path / "relocated-single"
    original.rename(moved)
    catalog = discovery.discover_evaluation_runs(tmp_path)
    assert len(catalog.runs) == 1
    run = catalog.runs[0]
    assert run.run_dir == moved
    assert run.config_label == "authored-experiment/fno_m2x2_h4_l1"


_HISTORY_IDENTITY = {"objective": {"id": "rmse", "direction": "minimize"}, "effective_config_digest": "d" * 64}
_HISTORY_CHECKPOINT = "c" * 64  # the selected checkpoint digest the fake lease admits
# (epoch, point, global step, objective, ends the epoch): a validation inside epoch 1 (eligible, not the best), one at its end
# (not eligible), and the budget end in epoch 2 (the best).
_VALIDATIONS = ((1, 5, 4.0, 0.5, False), (1, 10, 10.0, None, True), (2, 20, 20.0, 0.3, True))


def _admitted_history_run(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, *, task: str = "steady_flow", summary: dict[str, Any] | None = None, write: bool = True
) -> None:
    """Write a two-epoch history with the real writer and admit the run through a fake lease that yields its checkpoint evidence."""
    run_dir.mkdir()
    binding = {"task": task, "run_name": "run", "checkpoint_identity": _HISTORY_IDENTITY}
    if write:
        writer = training_history.make_epoch_state_callback(run_dir, **binding)
        for epoch, point, step, value, ends_epoch in _VALIDATIONS:
            validation = {
                "global_step": step,
                "validation/point": float(point),
                "validation/eligible": float(value is not None),
                **({} if value is None else {"id/rmse": value}),
            }
            writer(epoch, {"global_step": step, "train/loss_total": 1.0 / epoch} if ends_epoch else None, validation)
    admitted = {
        "run_dir": run_dir,
        "config": {"task": task, "run": {"name": "run"}, "model": {"kind": "fno"}, "loss": {"physics": {"enabled": True}}},
        "summary": {"task": task, **(summary or {})},
        "checkpoint_identity": _HISTORY_IDENTITY,
        "selected_checkpoint_sha256": _HISTORY_CHECKPOINT,
        "best_checkpoint": {"best_epoch": 2, "best_metric": 0.3},
        "last_checkpoint": {
            "completed_epoch": 2,
            "objective_history": [
                {"point": point, "epoch": epoch, "objective_id": "rmse", "value": value, "bound_violations": {}}
                for epoch, point, _step, value, _ends in _VALIDATIONS
            ],
        },
    }

    @contextmanager
    def lease(path: Path | str) -> Iterator[dict[str, Any]]:
        assert Path(path) == run_dir
        yield admitted

    monkeypatch.setattr(discovery.experiments.run, "saved_artifact_run_lease", lease)


def test_steady_stage_history_is_read_after_admission_with_its_selected_point(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A steady run's history comes back in order, at epoch 0, with the validation point of its best checkpoint."""
    run_dir = tmp_path / "steady"
    _admitted_history_run(run_dir, monkeypatch)

    history = discovery.stage_history(run_dir, expected_checkpoint_sha256=_HISTORY_CHECKPOINT)

    assert isinstance(history, discovery.StageHistory)
    assert (history.stage, history.pipeline, history.start_epoch, history.model_family) == ("steady", "run", 0, "pi-fno")
    assert (history.objective_id, history.objective_direction, history.selected_point) == ("rmse", "minimize", 20)
    assert [record["validation/point"] for record in history.validations] == [5.0, 10.0, 20.0]
    assert [record["epoch"] for record in history.epochs] == [1.0, 2.0]


def test_handoff_stage_history_starts_at_its_a0_epoch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A B or A1 stage belongs to its A0's pipeline and starts at the A0 handoff epoch."""
    run_dir = tmp_path / "b"
    summary = {"terminal_controller": {"arm": "B"}, "teacher_handoff": {"source_run_name": "a0-run", "best_epoch": 7}}
    _admitted_history_run(run_dir, monkeypatch, task="transient_drying", summary=summary)

    history = discovery.stage_history(run_dir, expected_checkpoint_sha256=_HISTORY_CHECKPOINT)

    assert isinstance(history, discovery.StageHistory)
    assert (history.stage, history.pipeline, history.start_epoch) == ("B", "a0-run", 7)


def test_a_stage_history_of_another_checkpoint_than_expected_raises_with_both_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The admitted run's selected checkpoint digest, read under the lease, must be the expected one; a mismatch is a refusal."""
    run_dir = tmp_path / "replaced"
    _admitted_history_run(run_dir, monkeypatch)

    with pytest.raises(RuntimeError, match=rf"{'e' * 64}.*{_HISTORY_CHECKPOINT}|{_HISTORY_CHECKPOINT}.*{'e' * 64}"):
        discovery.stage_history(run_dir, expected_checkpoint_sha256="e" * 64)


@pytest.mark.parametrize("defect", ["missing", "stale_value"])
def test_unusable_stage_history_is_unavailable_with_its_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    """No history, or one the validator refuses (here a stale objective at a non-selected validation), is unavailable, not an error."""
    run_dir = tmp_path / defect
    _admitted_history_run(run_dir, monkeypatch, write=defect != "missing")
    path = training_history.history_path(run_dir)
    if defect != "missing":
        persisted = json.loads(path.read_text(encoding="utf-8"))
        persisted["validations"][0]["id/rmse"] = 0.45
        path.write_text(json.dumps(persisted), encoding="utf-8")

    history = discovery.stage_history(run_dir, expected_checkpoint_sha256=_HISTORY_CHECKPOINT)

    assert isinstance(history, discovery.UnavailableHistory)
    assert history.run_dir == run_dir
    assert history.reason


def _study_payload(**changes: Any) -> dict[str, Any]:
    """Return a minimal study summary: one completed, one pruned and one failed trial."""
    trials = [
        {"number": 0, "status": "completed", "value": 0.4, "best_objective_so_far": 0.4},
        {"number": 1, "status": "pruned", "value": None, "best_objective_so_far": 0.4},
        {"number": 2, "status": "failed", "value": None, "best_objective_so_far": 0.4},
    ]
    payload = {
        "study_name": "study",
        "objective": {"id": "rmse", "direction": "minimize"},
        "trial_counts": {"completed": 1, "pruned": 1, "failed": 1, "diverged": 0},
        "trials": trials,
    }
    return {**payload, **changes}


def test_study_summary_reads_the_drawn_keys_and_binds_counts_to_trials(tmp_path: Path) -> None:
    """The reader returns the trials in order and refuses counts that disagree with their statuses."""
    path = tmp_path / "study_summary.json"
    path.write_text(json.dumps(_study_payload()), encoding="utf-8")
    summary = discovery.read_study_summary(path)
    assert [(trial.number, trial.status, trial.value) for trial in summary.trials] == [
        (0, "completed", 0.4),
        (1, "pruned", None),
        (2, "failed", None),
    ]
    assert summary.direction == "minimize"

    path.write_text(json.dumps(_study_payload(trial_counts={"completed": 2, "pruned": 1, "failed": 1})), encoding="utf-8")
    with pytest.raises(ValueError, match="trial_counts"):
        discovery.read_study_summary(path)


def test_study_summary_best_so_far_follows_completed_and_duplicate_trials_in_the_direction(tmp_path: Path) -> None:
    """A maximized study whose duplicate trial improves the best is read; the pruned trial's higher value counts for nothing."""
    trials = [
        {"number": 0, "status": "completed", "value": 0.4, "best_objective_so_far": 0.4},
        {"number": 1, "status": "pruned", "value": 0.9, "best_objective_so_far": 0.4},
        {"number": 2, "status": "duplicate", "value": 0.6, "best_objective_so_far": 0.6},
    ]
    path = tmp_path / "study_summary.json"
    counts = {"completed": 1, "pruned": 1, "duplicate": 1}
    payload = _study_payload(objective={"id": "r2", "direction": "maximize"}, trial_counts=counts, trials=trials)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert [trial.best_so_far for trial in discovery.read_study_summary(path).trials] == [0.4, 0.4, 0.6]


@pytest.mark.parametrize("defect", ["missing_value", "missing_best", "increasing_best", "pruned_best"])
def test_damaged_study_summary_is_refused(tmp_path: Path, defect: str) -> None:
    """Missing trial keys and a best so far that breaks the producer's status and direction rules are refused."""
    payload = _study_payload()
    trials = payload["trials"]
    if defect == "missing_value":
        del trials[1]["value"]
    elif defect == "missing_best":
        del trials[2]["best_objective_so_far"]
    elif defect == "increasing_best":
        trials[2]["best_objective_so_far"] = 0.5
    else:
        # A pruned trial's value never becomes the best.
        trials[1].update(value=0.1, best_objective_so_far=0.1)
    path = tmp_path / "study_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"value|best_objective_so_far"):
        discovery.read_study_summary(path)
