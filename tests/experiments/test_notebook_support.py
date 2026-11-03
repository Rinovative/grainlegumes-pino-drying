"""Protect lightweight, read-only notebook support services.

Responsibilities:
  - Check steady and transient plan previews, saved-run identity, and manifest-only reads
  - Check that explicit training Dataset and Stage-B seconds selections reach their stages and that only built Datasets of the task are listed
  - Execute maintained notebook inspection cells with a tiny test-owned configuration

Design principles:
  - Resolved context preserves scientific roles and measured-budget dependencies
  - Lightweight inspection cannot create training storage or load tensor artifacts

This module does NOT:
  - Train models or render evaluation plots
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from support import configs

from src import common, datasets, experiments

support = experiments.notebook_support


def test_prepare_context_resolves_synthetic_dataset_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve public notebook context for absent synthetic Dataset packages."""
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    config_path = configs.write_config(tmp_path, configs.direct_config())

    context = support.prepare_notebook_context(config_path)

    assert isinstance(context, support.NotebookContext)
    assert context.config_path == config_path
    assert context.task.id == "steady_flow"
    assert tuple(preview.dataset_id for preview in context.dataset_previews) == (
        "synthetic_train",
        "synthetic_ood",
    )
    assert all(not preview.metadata_validated for preview in context.dataset_previews)
    assert all(preview.path.is_relative_to(storage_root) for preview in context.dataset_previews)
    assert context.protocol == "direct"
    support.prepare_configuration_tables(context)
    support.prepare_run_preview_table(context, project_root=tmp_path, storage_root=storage_root)
    assert not storage_root.exists()


@pytest.mark.parametrize("matched", [False, True])
def test_explicit_train_dataset_selection_reaches_every_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    matched: bool,
) -> None:
    """Bind a notebook-selected Dataset and Stage-B seconds like the CLI when the YAML leaves them null."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "absent_storage"))
    raw = configs.transient_matched_config() if matched else configs.direct_config()
    raw["data"]["train_dataset"] = None
    if matched:
        raw["training"]["b"]["train_seconds"] = None
    path = configs.write_config(tmp_path, raw)

    seconds = 7.0 if matched else None
    with pytest.raises(ValueError, match="train_dataset"):
        support.prepare_notebook_context(path, train_seconds=seconds)
    context = support.prepare_notebook_context(path, train_dataset="selected", train_seconds=seconds)

    assert len(context.stage_configs) == (3 if matched else 1)
    assert {config["data"]["train_dataset"] for config in context.stage_configs} == {"selected"}
    assert context.dataset_previews[0].dataset_id == "selected"
    if matched:
        assert support.prepare_protocol_table(context).rows[1][4] == 7.0


def test_list_built_datasets_names_only_published_datasets_of_the_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List manifest-bearing Dataset folders of the configuration's task and fail when none exists."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    path = configs.write_config(tmp_path, configs.direct_config())
    with pytest.raises(FileNotFoundError, match="No built steady_flow Dataset"):
        support.list_built_datasets(path)

    root = common.paths.get_datasets_root()
    for task, dataset_id in (("steady_flow", "b-built"), ("steady_flow", "a-built"), ("transient_drying", "other-task")):
        manifest = common.paths.resolve_dataset_manifest_path(task, dataset_id, dataset_root=root)
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{}")
    common.paths.resolve_dataset_dir("steady_flow", "unpublished", dataset_root=root).mkdir()

    assert support.list_built_datasets(path) == ("a-built", "b-built")


@pytest.mark.parametrize("model_kind", ["fno", "uno", "rno"])
def test_transient_plan_preview_preserves_roles_and_stage_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_kind: str,
) -> None:
    """Preview supported transient architectures without loading datasets or models."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "absent_storage"))
    raw = configs.transient_matched_config(model_kind=model_kind)
    path = configs.write_config(tmp_path, raw)
    context = support.prepare_notebook_context(path, stage="b")

    assert context.protocol == "matched_a1_b"
    assert context.official_config["training"]["comparison_arm"] == "b"
    assert tuple(preview.role for preview in context.dataset_previews) == (
        "ID train / validation / test source",
        "Dataset-declared OOD",
    )
    assert context.stage_configs[1]["training"]["teacher_handoff"]["source_run_name"] == context.stage_configs[0]["run"]["name"]
    tables = support.prepare_configuration_tables(context)
    settings = {row[0]: row[1] for table in tables for row in table.rows}
    assert settings["Time conditioning"] == "kind=none"
    sampling = settings["Sampling"]
    assert isinstance(sampling, str)
    assert sampling.startswith("mode=trajectory" if model_kind == "rno" else "mode=rollout_window")
    assert settings["Kind"] == model_kind
    assert settings["Primary objective"] == "normalized_drying_group_macro_rmse"
    assert "Physics enabled" not in settings
    support.prepare_run_preview_table(context, project_root=tmp_path, storage_root=tmp_path)
    assert not (tmp_path / "absent_storage").exists()
    with pytest.raises(ValueError, match="unavailable"):
        support.prepare_notebook_context(path, stage="a2")


def test_matched_plan_preview_requires_no_additional_evaluation_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind A1 to measured B evidence while preserving explicit empty additional roles."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "absent_storage"))
    raw = configs.transient_matched_config()
    raw["data"]["additional_evaluation_datasets"] = []
    path = configs.write_config(tmp_path, raw)
    context = support.prepare_notebook_context(path, stage="a1")
    assert context.protocol == "matched_a1_b"
    assert len(context.dataset_previews) == 1
    protocol = support.prepare_protocol_table(context)
    assert tuple(row[0] for row in protocol.rows) == ("a0", "b", "a1")
    assert protocol.rows[1][4] == 0.25
    assert protocol.rows[2][4] is None
    assert protocol.rows[2][5] == context.stage_configs[1]["run"]["name"]
    assert protocol.rows[1][3] == protocol.rows[2][3] == protocol.rows[0][1]
    support.prepare_configuration_tables(context)
    assert not context.run_dir.exists()


def test_saved_transient_run_inspection_uses_its_own_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect a persisted transient leaf with no invented OOD or model loading."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    raw = configs.transient_matched_config()
    raw["data"]["additional_evaluation_datasets"] = []
    path = configs.write_config(tmp_path, raw)
    config = support.prepare_notebook_context(path).official_config
    run_dir = tmp_path / "saved_run"
    configs.write_yaml(run_dir / "config.yaml", config)
    summary = {
        "schema_version": 1,
        "task": "transient_drying",
        "run_name": config["run"]["name"],
        "status": "failed",
        "completed_epoch": 1,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    inspection = support.prepare_run_inspection(run_dir)
    assert dict(inspection.summary_rows)["Status"] == "failed"
    assert dict(inspection.summary_rows)["Objective"] == "normalized_drying_group_macro_rmse"
    paths = dict(inspection.existence_rows)
    assert "history.json" in paths
    assert paths["history.json"] is False
    assert not any("ood" in label or "additional_evaluation" in label for label in paths)
    summary["run_name"] = "mismatched"
    (run_dir / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="identity disagree"):
        support.prepare_run_inspection(run_dir)


def test_training_notebook_executes_inspection_only_for_a_test_owned_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run every notebook cell against a test-owned project without touching storage."""
    project = tmp_path / "project"
    monkeypatch.setenv("PROJECT_ROOT", str(project))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    raw = configs.transient_matched_config()
    raw["data"]["additional_evaluation_datasets"] = []
    configs.write_config(project, raw, canonical_root=project)
    monkeypatch.setattr(support, "list_built_datasets", lambda _config_path: ("selected_dataset",))
    notebook = json.loads((Path(__file__).resolve().parents[2] / "notebooks/03_training_pipeline.ipynb").read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"]).replace("TRAIN_SECONDS = None", "TRAIN_SECONDS = 7.0")
            exec(compile(source, f"03_training_pipeline:cell{index}", "exec"), namespace)  # noqa: S102 -- execute only maintained notebook code

    assert namespace["CONTEXT"].task.id == "transient_drying"
    assert namespace["CONTEXT"].protocol == "matched_a1_b"
    assert {config["data"]["train_dataset"] for config in namespace["CONTEXT"].stage_configs} == {"selected_dataset"}
    assert support.prepare_protocol_table(namespace["CONTEXT"]).rows[1][4] == 7.0
    assert "--train-dataset selected_dataset" in namespace["train_command"]
    assert "--train-seconds 7.0" in namespace["train_command"]
    assert not (tmp_path / "storage").exists()


def test_metadata_preview_admits_manifest_without_reading_tensor_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep metadata-only previews distinct from a full payload integrity check."""
    from support import synthetic_task  # noqa: PLC0415 -- only this test needs selection provenance

    storage = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage))
    provenance = synthetic_task.build_synthetic_selection_provenance(dataset_name="preview", sample_ids=["case-1", "case-2"])
    from src.datasets.packages import dataset_packages_reproduction as reproduction  # noqa: PLC0415

    provenance["builder_identity"]["implementation_sha256"] = reproduction.conversion_implementation_sha256("steady_flow")
    dataset_id, digest = datasets.contracts.identity.package_identity_from_provenance(provenance)
    dataset_root = common.paths.get_datasets_root()
    payload = common.paths.resolve_dataset_path("steady_flow", dataset_id, dataset_root=dataset_root)
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"not a tensor; preview must not deserialize or hash payload bytes")
    manifest = {
        **provenance,
        "dataset_id": dataset_id,
        "dataset_digest": digest,
        "payload_filename": payload.name,
        "sample_count": 2,
        "source_case_count": 2,
        "transition_count": 0,
        "payload_sha256": "a" * 64,
    }
    manifest_path = payload.parent / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    raw = configs.direct_config()
    raw["data"]["train_dataset"] = dataset_id
    path = configs.write_config(tmp_path, raw)
    context = support.prepare_notebook_context(path)
    preview = context.dataset_previews[0]
    assert preview.metadata_validated is True
    assert preview.sample_count == 2
    assert preview.dataset_digest == digest
    assert preview.path == payload
    manifest["channel_contract_digest"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract digest"):
        support.prepare_notebook_context(path)
