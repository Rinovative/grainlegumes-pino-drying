"""Protect current artifact reader identity, cache admission, and contained rebuilds.

Responsibilities:
  - Check table schema, metadata identity, contained rebuilds, and run discovery admission
  - Check that a recorded analysis producer is identity evidence, validated strictly and never compared with the running code

Design principles:
  - Scientific cache identity excludes runtime location; unsafe paths fail before mutation

This module does NOT:
  - Verify numerical artifact generation or plot usability
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from src import analysis, common, experiments
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract

if TYPE_CHECKING:
    from pathlib import Path

_TREE_ID = "a" * 12 + "-" + "b" * 12


def _current_generic_frame(*, metadata: object, index: int = 0, source_index: int = 2) -> pd.DataFrame:
    """
    Build one compact current-schema artifact table.

    ``metadata`` is intentionally left unvalidated so callers can exercise the
    reader's JSON, collision, and alignment checks against one canonical row.
    """
    return pd.DataFrame(
        [
            {
                "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
                "task_id": "synthetic",
                "output_fields": ["target"],
                "output_units": ["1"],
                "case_index": source_index + 1,
                "source_index": source_index,
                "split_local_index": 0,
                "source_dataset_id": "dataset",
                "npz_path": "case_0003.npz",
                "meta": metadata,
                "inference_time_ms": None,
                "rel_l2": 0.0,
                "rel_h1": 0.0,
                "physical_sse_target": 0.0,
                "physical_count_target": 1,
                "physical_rmse_target": 0.0,
                "normalized_sse_target": 0.0,
                "normalized_count_target": 1,
                "normalized_rmse_target": 0.0,
                "prediction_validity": json.dumps(validity_contract.build_field_prediction_validity(np.zeros((1, 1, 1)), channels=("target",))),
            }
        ],
        index=[index],
    )


def test_dataframe_reader_admits_current_sufficient_statistics() -> None:
    """
    Admit exact physical and normalized evidence without inventing provenance.

    A current one-row table exposes its schema and field contract while remaining
    explicitly table-only until authoritative group and scale provenance is bound.
    """
    enriched = analysis.evaluation.dataframe.build_eval_df(_current_generic_frame(metadata='{"label": "valid"}'))

    assert enriched.attrs["artifact_schema_version"] == analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION
    assert enriched.attrs["output_fields"] == ("target",)
    assert enriched.attrs["provenance_complete"] is False
    assert analysis.evaluation.dataframe.PRIMARY_OBJECTIVE_ID not in enriched.attrs
    assert enriched.loc[0, "physical_rmse_target"] == 0.0


def test_dataframe_reader_rejects_every_unexpected_column() -> None:
    """Reject an undeclared column through the generic exact-schema boundary."""
    frame = _current_generic_frame(metadata="{}")
    frame["unexpected_metric"] = 0.0

    with pytest.raises(ValueError, match="schema mismatch"):
        analysis.evaluation.dataframe.build_eval_df(frame)


@pytest.mark.parametrize(
    "schema_version",
    [float(analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION), analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION + 1],
    ids=("floating-current", "other-version"),
)
def test_dataframe_reader_requires_the_current_integer_artifact_version(schema_version: float) -> None:
    """Reject alternate representations and unsupported Parquet schema versions."""
    frame = _current_generic_frame(metadata="{}")
    frame["artifact_schema_version"] = schema_version  # pyright: ignore[reportCallIssue, reportArgumentType]

    with pytest.raises((TypeError, ValueError), match=r"artifact_schema_version|schema version"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_dataframe_reader_rejects_missing_and_duplicate_columns() -> None:
    """Reject missing evidence and duplicate columns through exact schema admission."""
    missing = _current_generic_frame(metadata="{}").drop(columns="normalized_sse_target")
    with pytest.raises(ValueError, match="schema mismatch"):
        analysis.evaluation.dataframe.build_eval_df(missing)

    base = _current_generic_frame(metadata="{}")
    duplicate = pd.concat([base, base[["physical_rmse_target"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns"):
        analysis.evaluation.dataframe.build_eval_df(duplicate)


def test_metadata_cannot_duplicate_authoritative_identity_columns() -> None:
    """
    Reject metadata that duplicates an authoritative identity column.

    A forged ``source_index`` must fail before flattening so metadata cannot
    override the row identity used for provenance and case navigation.
    """
    frame = _current_generic_frame(metadata='{"source_index": 999}')

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_cannot_reuse_the_raw_meta_column() -> None:
    """
    Reject metadata that reuses the raw ``meta`` column name.

    The reader must preserve the original payload boundary rather than allow a
    flattened key to erase or ambiguously replace its own source column.
    """
    frame = _current_generic_frame(metadata='{"meta": "ambiguous"}')

    with pytest.raises(ValueError, match="collides with authoritative"):
        analysis.evaluation.dataframe.build_eval_df(frame)


def test_metadata_expansion_preserves_noncontiguous_row_indices() -> None:
    """
    Preserve metadata alignment for a noncontiguous DataFrame index.

    A row indexed at seven must retain its case identity and label after
    expansion. Positional concatenation would attach metadata to the wrong row.
    """
    row_index = 7
    case_index = 8
    frame = _current_generic_frame(
        metadata='{"label": "case-eight"}',
        index=row_index,
        source_index=case_index - 1,
    )

    enriched = analysis.evaluation.dataframe.build_eval_df(frame)

    assert enriched.index.tolist() == [row_index]
    assert enriched.loc[row_index, "case_index"] == case_index
    assert enriched.loc[row_index, "label"] == "case-eight"


@pytest.mark.parametrize(
    ("metadata", "error_type", "match"),
    [
        ("{'source': 1}", ValueError, "valid JSON"),
        ("[1, 2]", TypeError, "decode to an object"),
        (42, TypeError, "JSON object or mapping"),
    ],
    ids=("python-literal", "json-array", "non-string"),
)
def test_artifact_metadata_requires_a_json_object(
    metadata: object,
    error_type: type[Exception],
    match: str,
) -> None:
    """
    Reject malformed JSON, JSON arrays, and unsupported metadata values.

    The parameter matrix covers syntax, decoded shape, and input type so the
    reader never turns an invalid provenance payload into an empty mapping.
    """
    with pytest.raises(error_type, match=match):
        analysis.evaluation.dataframe.build_eval_df(_current_generic_frame(metadata=metadata))


def _identity_provenance() -> dict[str, Any]:
    """Return one complete steady provenance document carrying every identity section."""
    return {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "producer": {"analysis_sha256": "e" * 64},
        "run": {
            "name": "run-a",
            "task": "synthetic",
            "task_contract_digest": "task-a",
            "effective_config_digest": "config-a",
            "best_checkpoint_sha256": "checkpoint-a",
            "normalizer_sha256": "normalizer-a",
        },
        "model": {
            "kind": "fno",
            "architecture": {"hidden_channels": 8},
            "physics_enabled": True,
            "parameter_counts": {"total": 10, "trainable": 10},
        },
        "split_role": "eval",
        "dataset": {
            "name": "dataset-a",
            "role": "id",
            "source_dataset_ids": ["dataset-a"],
            "source_identities": [{"dataset_id": "dataset-a", "dataset_digest": "data-a"}],
            "case_ids_by_source": [["case-a", "case-b"]],
            "membership_digests": ["membership-a"],
            "evaluation_assignments": [],
        },
        "selection": {
            "full_selected_case_count": 2,
            "effective_case_count": 2,
            "generation_limit": None,
            "full_ordered_source_indices_sha256": "full-membership-a",
            "effective_ordered_source_indices_sha256": "membership-a",
        },
        "normalizer": {
            "sha256": "normalizer-a",
            "fit_split": "train",
            "output_normalization": "per_channel_standardization",
            "output_standard_deviations": {"first": 1.0, "second": 2.0},
            "denominator_floor": 1.0e-6,
        },
        "evaluator": {
            "metrics": [{"id": "rel_l2", "fields": ["first", "second"]}],
            "objective": {
                **analysis.evaluation.dataframe.PRIMARY_OBJECTIVE_DEFINITION,
                "fields": "all",
            },
            "input_fields": ["input"],
            "input_units": {"input": "1"},
            "output_fields": ["first", "second"],
            "output_units": {"first": "1", "second": "1"},
            "output_groups": [
                {"id": "first_group", "fields": ["first"]},
                {"id": "second_group", "fields": ["second"]},
            ],
            "physics_kind": "steady_brinkman",
            "group_objective_evidence": {"reduction": "equal_group_mean"},
            "predictive_metrics": {"rel_l2": "per_case"},
        },
        "physics": {
            "residual_schema_version": 1,
            "task_id": "synthetic",
            "task_contract_digest": "task-a",
            "equation_kind": "steady_brinkman",
            "equation_set": ["momentum", "continuity"],
            "continuity_kind": "cross_section_flux_balance",
            "boundary_condition_kind": "pressure_drop",
            "scales": {"pressure": {"formula": "mean_inlet(p_in_bc)", "unit": "Pa"}},
            "derivatives": {"kind": "physical"},
            "interior_crop": 2,
            "residual_evaluation_region": {"kind": "interior"},
            "constants": {"dynamic_viscosity_pa_s": 1.8139e-5},
            "permeability_representation": {"kind": "symmetric_tensor"},
            "scalar_definitions": {"flux_balance_mse": "mean(flux_balance**2)"},
            "array_definitions": {"flux_balance": "(q(y) - mean_y(q)) / flux"},
        },
        "generation": {
            "effective_case_limit": None,
            "inference_batch_size": 1,
            "compression": "numpy savez_compressed",
        },
        "runtime": {"requested_policy": "cpu", "resolved_device": "cpu", "inference_batch_size": 1, "source_tree_id": None},
        "timing": {"report": {"measured": False}, "operational_performance": {"total_seconds": 1.0}},
    }


def test_artifact_identity_tracks_science_but_not_runtime_device() -> None:
    """
    Separate scientific cache identity from operational runtime facts.

    Representative producer, task, training, data, checkpoint, evaluation, and physics
    changes alter identity. Runtime device and locator facts do not.
    """
    provenance = _identity_provenance()
    baseline = analysis.artifacts.contracts.artifact_identity_digest(provenance)
    mutations = [
        (("artifact_schema_version",), 999),
        (("producer", "analysis_sha256"), "f" * 64),
        (("run", "effective_config_digest"), "config-b"),
        (("run", "best_checkpoint_sha256"), "checkpoint-b"),
        (("dataset", "source_identities", 0, "dataset_digest"), "data-b"),
        (("selection", "effective_ordered_source_indices_sha256"), "membership-b"),
        (("evaluator", "objective", "reduction"), "wrong-reduction"),
        (("physics", "continuity_kind"), "pointwise_divergence"),
        (("physics", "scales", "pressure", "formula"), "max_inlet(p_in_bc)"),
        (("generation", "inference_batch_size"), 2),
    ]
    for keys, replacement in mutations:
        changed = copy.deepcopy(provenance)
        target: object = changed
        for key in keys[:-1]:
            assert isinstance(target, (dict, list))
            target = target[key]
        assert isinstance(target, (dict, list))
        target[keys[-1]] = replacement
        assert analysis.artifacts.contracts.artifact_identity_digest(changed) != baseline

    operational = copy.deepcopy(provenance)
    operational["runtime"] = {
        "requested_policy": "auto",
        "resolved_device": "cuda:7",
        "inference_batch_size": 19,
        "source_tree_id": _TREE_ID,
    }
    operational["timing"] = {"report": {"measured": True}, "operational_performance": {"total_seconds": 3.0}}
    operational["run"]["name"] = "renamed-run"
    operational["dataset"]["source_path"] = "/relocated/dataset.pt"
    assert analysis.artifacts.contracts.artifact_identity_digest(operational) == baseline


def test_recorded_analysis_producer_is_admitted_as_evidence_and_never_compared_with_the_running_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """A read after an analysis-code change keeps the identity; a missing, malformed or extended producer section is refused."""
    contracts = analysis.artifacts.contracts
    provenance = _identity_provenance()
    baseline = contracts.artifact_identity_digest(provenance)
    for task_id in contracts.ANALYSIS_PROJECTIONS:
        changed = SimpleNamespace(name=f"analysis_{task_id}", sha256=lambda: "0" * 64)
        monkeypatch.setitem(contracts.ANALYSIS_PROJECTIONS, task_id, changed)
    assert contracts.artifact_identity_digest(provenance) == baseline
    for producer in (None, {}, {"analysis_sha256": "E" * 64}, {"analysis_sha256": "e" * 63}, {"analysis_sha256": "e" * 64, "extra": 1}):
        malformed = {**copy.deepcopy(provenance), "producer": producer}
        with pytest.raises(ValueError, match="producer must be exactly"):
            contracts.artifact_identity_digest(malformed)
    with pytest.raises(ValueError, match=r"missing semantic sections \['producer'\]"):
        contracts.artifact_identity_digest({key: value for key, value in provenance.items() if key != "producer"})


@pytest.mark.parametrize("tree_id", [None, _TREE_ID])
def test_runtime_provenance_records_the_executing_source_tree(tree_id: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both tasks persist the executing frozen tree as a required runtime fact; the validator admits exactly null or a tree name."""
    contracts = analysis.artifacts.contracts
    monkeypatch.setattr(common.paths, "source_tree_id", lambda *_args, **_kwargs: tree_id)
    device = {"requested_policy": "cpu", "resolved_device": "cpu", "device_type": "cpu", "pytorch_version": "test"}
    runtime = contracts.runtime_provenance(device, inference_batch_size=2)

    assert runtime["source_tree_id"] == tree_id
    assert contracts.validate_runtime_provenance(runtime) == 2
    without = {key: value for key, value in runtime.items() if key != "source_tree_id"}
    with pytest.raises(ValueError, match=r"missing=\['source_tree_id'\]"):
        contracts.validate_runtime_provenance(without)
    for malformed in ("", _TREE_ID.upper(), _TREE_ID[:-1], 7):
        with pytest.raises(ValueError, match="source_tree_id must be null or"):
            contracts.validate_runtime_provenance({**runtime, "source_tree_id": malformed})


def test_artifact_target_rejects_broad_and_out_of_run_paths(tmp_path: Path) -> None:
    """Reject the analysis root, the additional-evaluation parent and out-of-run paths."""
    run_dir = tmp_path / "run"
    for save_root in (run_dir / "analysis", run_dir / "analysis" / "additional_evaluation", tmp_path / "outside"):
        with pytest.raises(ValueError, match="exact artifact target"):
            analysis.artifacts.service._validated_artifact_target(run_dir=run_dir, save_root=save_root)


def test_upload_gate_requires_an_explicit_complete_current_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The upload gate returns what the shared role validator admits and refuses what it refuses."""
    from src.analysis.evaluation import evaluation_artifact_loader as loader  # noqa: PLC0415

    run_dir = tmp_path / "run"
    artifact_root = common.paths.resolve_id_analysis_dir(run_dir)
    artifact_root.mkdir(parents=True)
    service = analysis.artifacts.service
    monkeypatch.setattr(service.experiments.run, "validate_completed_run", lambda _run_dir: {"config": {}})
    monkeypatch.setattr(service.experiments.config.loader, "validate_resolved_task_contract", lambda _config: SimpleNamespace(id="steady_flow"))
    monkeypatch.setattr(
        analysis.artifacts.plan,
        "artifact_plan_from_completed",
        lambda *_args, **_kwargs: SimpleNamespace(id_role=SimpleNamespace(split="eval", dataset_name="dataset"), additional_role=None),
    )
    frame = pd.DataFrame()
    frame.attrs["artifact_provenance"] = {"artifact_kind": "steady_field"}
    calls: list[dict[str, Any]] = []

    def admit(_run_dir: Path, _completed: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(frame=frame)

    monkeypatch.setattr(loader, "admit_role_artifact", admit)
    assert service.validate_artifact_upload_source(run_dir=run_dir, artifact_root=artifact_root) == {"artifact_kind": "steady_field"}
    assert calls == [{"selection_role": "id", "root": artifact_root.resolve(), "evaluation_spatial_stride": 1}]

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        msg = "Artifact output manifest does not match the current payload files."
        raise ValueError(msg)

    monkeypatch.setattr(loader, "admit_role_artifact", refuse)
    with pytest.raises(service.ArtifactCacheError, match="output manifest"):
        service.validate_artifact_upload_source(run_dir=run_dir, artifact_root=artifact_root)


def test_rebuild_rejects_symlink_escape(tmp_path: Path) -> None:
    """
    Reject analysis-root and target symlinks that escape containment.

    Each attempted rebuild points through a different escape shape, and every
    outside or sibling marker must remain intact after rejection.
    """
    run_dir = tmp_path / "run"
    outside = tmp_path / "outside"
    outside_target = outside / "id"
    outside_target.mkdir(parents=True)
    marker = outside_target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    run_dir.mkdir()
    (run_dir / "analysis").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service._validated_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "id")

    assert marker.read_text(encoding="utf-8") == "keep"

    target_symlink_run = tmp_path / "target-symlink-run"
    (target_symlink_run / "analysis").mkdir(parents=True)
    (target_symlink_run / "analysis" / "id").symlink_to(outside_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service._validated_artifact_target(
            run_dir=target_symlink_run,
            save_root=target_symlink_run / "analysis" / "id",
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    sibling_run = tmp_path / "sibling-symlink-run"
    sibling_target = sibling_run / "analysis" / "additional_evaluation" / "keep"
    sibling_target.mkdir(parents=True)
    sibling_marker = sibling_target / "keep.txt"
    sibling_marker.write_text("keep", encoding="utf-8")
    (sibling_run / "analysis" / "id").symlink_to(sibling_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service._validated_artifact_target(
            run_dir=sibling_run,
            save_root=sibling_run / "analysis" / "id",
        )
    assert sibling_marker.read_text(encoding="utf-8") == "keep"


def test_requested_run_names_cannot_escape_discovery_root(tmp_path: Path) -> None:
    """
    Reject requested run names that escape the discovery root.

    Treating a traversal string as a logical run name must fail before directory
    discovery, preventing explicit selection from bypassing containment.
    """
    with pytest.raises(ValueError, match="single non-empty path component"):
        list(analysis.artifacts.service.iter_run_dirs(tmp_path, run_names=["../outside"]))


@pytest.mark.parametrize("marker_name", common.paths.CURRENT_RUN_REQUIRED_FILES)
def test_discovery_rejects_every_malformed_child_run_marker(
    tmp_path: Path,
    marker_name: str,
) -> None:
    """
    Reject a child containing any lone required run marker.

    The parameter covers every current lifecycle marker. Discovery must surface
    each partial run as corruption instead of silently omitting it.
    """
    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / marker_name).touch()

    with pytest.raises(experiments.run.RunLifecycleError, match=r"evaluation evidence|best checkpoint|invalid JSON"):
        list(analysis.artifacts.service.iter_run_dirs(tmp_path))


def test_discovery_skips_unselected_non_completed_runs_and_refuses_a_selected_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-completed run never blocks the completed siblings of a root, yet an explicit name selection stays strict."""
    service = analysis.artifacts.service
    for name, status in (("finished", "completed"), ("unfinished", "interrupted")):
        run = tmp_path / name
        run.mkdir()
        (run / "config.yaml").write_text(json.dumps({"run": {"name": name}}), encoding="utf-8")
        (run / "summary.json").write_text(json.dumps({"schema_version": 1, "status": status}), encoding="utf-8")
    admit = service.experiments.run.validate_evaluable_run
    monkeypatch.setattr(
        service.experiments.run,
        "validate_evaluable_run",
        lambda run: {"scientific_run_name": run.name} if run.name == "finished" else admit(run),
    )

    assert list(service.iter_run_dirs(tmp_path)) == [tmp_path / "finished"]
    skipped = capsys.readouterr().out
    assert "unfinished" in skipped
    assert "interrupted" in skipped
    assert list(service.iter_run_dirs(tmp_path, run_names=["finished"])) == [tmp_path / "finished"]
    with pytest.raises(experiments.run.RunLifecycleError, match=r"interrupted.*Resume the run to completion"):
        list(service.iter_run_dirs(tmp_path, run_names=["unfinished"]))


_TRIAL_LAYOUT = {
    "task": "transient_drying",
    "dataset": "dataset-id",
    "family": "optuna",
    "config_name": "fno_study",
    "model_label": "fno_m2x2_h4_l1",
    "seed": 17,
    "revision": None,
    "budget_epochs": 3,
    "stage": "single",
    "pair": None,
    "trial": "0001__fno_m2x2_h4_l1__lr-0p001",
    "a0_id": "a" * 16,
    "branch_id": None,
}


def test_discovery_skips_valid_optuna_trials_but_rejects_malformed_layouts_and_normal_runs_without_checkpoints(tmp_path: Path) -> None:
    """Skip a finished trial (config and summary only) by its valid recorded layout; keep failing closed otherwise."""

    def leaf(root: Path, **changes: object) -> Path:
        directory = root / "run"
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(json.dumps({"run": {"layout": {**_TRIAL_LAYOUT, **changes}}}), encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps({"schema_version": 1, "status": "completed"}), encoding="utf-8")
        return root

    assert list(analysis.artifacts.service.iter_run_dirs(leaf(tmp_path / "trial"))) == []
    with pytest.raises(ValueError, match="trial"):
        list(analysis.artifacts.service.iter_run_dirs(leaf(tmp_path / "malformed", trial=False)))
    normal = leaf(tmp_path / "normal", family="base", budget_epochs=None, trial=None)
    with pytest.raises(experiments.run.RunLifecycleError, match=r"evaluation evidence|best checkpoint"):
        list(analysis.artifacts.service.iter_run_dirs(normal))


def test_artifacts_refuse_an_explicitly_selected_optuna_trial_before_reading_any_checkpoint(tmp_path: Path) -> None:
    """A trial leaf chosen as the target gets one clear refusal naming the supported targets, on the discovery and the scoped path."""
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "config.yaml").write_text(json.dumps({"run": {"layout": _TRIAL_LAYOUT}}), encoding="utf-8")
    (trial / "summary.json").write_text(json.dumps({"schema_version": 1, "status": "completed"}), encoding="utf-8")
    refusal = r"Optuna trial.*no artifacts.*parent study.*training run"

    with pytest.raises(ValueError, match=refusal):
        list(analysis.artifacts.service.iter_run_dirs(trial))
    with pytest.raises(ValueError, match=refusal):
        analysis.artifacts.service.build_scoped_transient_artifact(
            run_dir=trial, dataset_root=tmp_path, output_root=tmp_path / "scoped", split="id", one_case=True
        )
    assert list(analysis.artifacts.service.iter_run_dirs(tmp_path)) == []
