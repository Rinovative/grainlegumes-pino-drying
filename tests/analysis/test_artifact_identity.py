# ruff: noqa: S101
"""
Protect current artifact reader identity, cache admission, and contained rebuilds.

The tests cover current table parsing, metadata collision rejection, ordered
membership, symlink/path containment, concurrent publication, and provenance
completion races. Numerical artifact generation is covered in
``test_artifact_provenance``. Plot usability is covered separately.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from src import analysis, common, experiments

if TYPE_CHECKING:
    from pathlib import Path


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
    [True, 1.0, 2],
    ids=("boolean-one", "floating-one", "unsupported-integer"),
)
def test_dataframe_reader_requires_integer_artifact_version_one(schema_version: bool | float) -> None:
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


def test_artifact_identity_tracks_science_but_not_runtime_device() -> None:
    """
    Separate scientific cache identity from operational runtime facts.

    Representative task, training, data, checkpoint, evaluation, and physics
    changes alter identity. Runtime device and locator facts do not.
    """
    provenance = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "run": {
            "name": "run-a",
            "task": "synthetic",
            "task_contract_digest": "task-a",
            "effective_config_digest": "config-a",
            "best_checkpoint_sha256": "checkpoint-a",
            "normalizer_sha256": "normalizer-a",
            "lifecycle_status": "completed",
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
            "full_case_count": 2,
            "fingerprint": "dataset-a",
            "data_contract_digest": "data-a",
            "saved_membership_digest": "membership-a",
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
            "boundary_condition_kind": "pressure_drop",
            "selected_training_continuity": "div_eps_velocity",
            "evaluated_continuity_formulations": ["div_velocity", "div_eps_velocity"],
            "derivatives": {"kind": "spectral", "extension": "reflect"},
            "interior_crop": 2,
            "residual_evaluation_region": {"kind": "interior"},
            "constants": {"dynamic_viscosity_pa_s": 1.8139e-5},
            "permeability_representation": {"kind": "symmetric_tensor"},
            "scalar_definitions": {"div_velocity_mse": "mean(div(u)**2)"},
            "array_definitions": {"div_u": "du/dx + dv/dy"},
        },
        "generation": {
            "effective_case_limit": None,
            "inference_batch_size": 1,
            "compression": "numpy savez_compressed",
        },
        "runtime": {"requested_policy": "cpu", "resolved_device": "cpu", "batch_size": 1},
    }
    baseline = analysis.artifacts.contracts.artifact_identity_digest(provenance)
    mutations = [
        (("artifact_schema_version",), 999),
        (("run", "effective_config_digest"), "config-b"),
        (("run", "best_checkpoint_sha256"), "checkpoint-b"),
        (("dataset", "fingerprint"), "dataset-b"),
        (("selection", "effective_ordered_source_indices_sha256"), "membership-b"),
        (("evaluator", "objective", "reduction"), "wrong-reduction"),
        (("physics", "selected_training_continuity"), "div_velocity"),
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
        "batch_size": 19,
    }
    operational["run"]["name"] = "renamed-run"
    operational["dataset"]["source_path"] = "/relocated/dataset.pt"
    assert analysis.artifacts.contracts.artifact_identity_digest(operational) == baseline


def test_rebuild_removes_only_one_exact_target(tmp_path: Path) -> None:
    """
    Confine rebuild deletion to one exact ID or named OOD target.

    Removing one OOD fixture must preserve the ID and sibling OOD markers, and
    broad or out-of-run paths must be rejected before filesystem mutation.
    """
    run_dir = tmp_path / "run"
    id_target = run_dir / "analysis" / "id"
    first_ood = run_dir / "analysis" / "ood" / "first"
    second_ood = run_dir / "analysis" / "ood" / "second"
    for target in (id_target, first_ood, second_ood):
        target.mkdir(parents=True)
        (target / "marker").write_text(target.name, encoding="utf-8")

    analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=first_ood)

    assert not first_ood.exists()
    assert (id_target / "marker").is_file()
    assert (second_ood / "marker").is_file()
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "ood")
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=tmp_path / "outside")


def test_upload_gate_requires_an_explicit_complete_current_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Admit only an explicit, complete, current artifact for upload.

    The valid fixture must pass without changing its bytes. After the manifest
    is made stale, validation must fail before any tracking-side effect.
    """
    run_dir = tmp_path / "run"
    artifact_root = common.paths.resolve_id_analysis_dir(run_dir)
    artifact_root.mkdir(parents=True)
    marker = artifact_root / "aggregate.parquet"
    marker.write_bytes(b"complete-local-artifact")
    provenance_path = analysis.artifacts.contracts.artifact_provenance_path(artifact_root)
    provenance_path.write_text("{}\n", encoding="utf-8")
    outputs = {"aggregate.parquet": {"sha256": "a" * 64, "size_bytes": 23}}
    provenance = {
        "provenance_schema_version": analysis.artifacts.contracts.ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "artifact_schema_version": analysis.artifacts.contracts.ARTIFACT_SCHEMA_VERSION,
        "outputs": outputs,
        "run": {
            "name": "run-name",
            "task": "steady_flow",
            "task_contract_digest": "t" * 64,
            "effective_config_digest": "c" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "normalizer_sha256": "n" * 64,
        },
    }
    completed = {
        "summary": {
            "effective_config_digest": "c" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "normalizer_sha256": "n" * 64,
        },
        "config": {"run": {"name": "run-name"}},
    }
    monkeypatch.setattr(
        analysis.artifacts.service,
        "_read_artifact_provenance",
        lambda _path: provenance,
    )
    monkeypatch.setattr(
        analysis.artifacts.service,
        "_require_current_provenance_schema",
        lambda _provenance: None,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.contracts,
        "artifact_output_manifest",
        lambda _root: outputs,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.run,
        "validate_completed_run",
        lambda _run_dir: completed,
    )
    monkeypatch.setattr(
        analysis.artifacts.service.experiments.config.loader,
        "validate_resolved_task_contract",
        lambda _config: SimpleNamespace(
            id="steady_flow",
            contract_digest="t" * 64,
        ),
    )

    validated = analysis.artifacts.service.validate_artifact_upload_source(
        run_dir=run_dir,
        artifact_root=artifact_root,
    )

    assert validated == provenance
    assert marker.read_bytes() == b"complete-local-artifact"
    monkeypatch.setattr(
        analysis.artifacts.service.contracts,
        "artifact_output_manifest",
        lambda _root: {"aggregate.parquet": {"sha256": "stale"}},
    )
    with pytest.raises(analysis.artifacts.service.ArtifactCacheError, match="manifest mismatch"):
        analysis.artifacts.service.validate_artifact_upload_source(
            run_dir=run_dir,
            artifact_root=artifact_root,
        )
    assert marker.read_bytes() == b"complete-local-artifact"


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
        analysis.artifacts.service.rebuild_artifact_target(run_dir=run_dir, save_root=run_dir / "analysis" / "id")

    assert marker.read_text(encoding="utf-8") == "keep"

    target_symlink_run = tmp_path / "target-symlink-run"
    (target_symlink_run / "analysis").mkdir(parents=True)
    (target_symlink_run / "analysis" / "id").symlink_to(outside_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(
            run_dir=target_symlink_run,
            save_root=target_symlink_run / "analysis" / "id",
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    sibling_run = tmp_path / "sibling-symlink-run"
    sibling_target = sibling_run / "analysis" / "ood" / "keep"
    sibling_target.mkdir(parents=True)
    sibling_marker = sibling_target / "keep.txt"
    sibling_marker.write_text("keep", encoding="utf-8")
    (sibling_run / "analysis" / "id").symlink_to(sibling_target, target_is_directory=True)
    with pytest.raises(ValueError, match="exact artifact target"):
        analysis.artifacts.service.rebuild_artifact_target(
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

    with pytest.raises(experiments.run.RunLifecycleError, match=r"evaluation evidence|best checkpoint"):
        list(analysis.artifacts.service.iter_run_dirs(tmp_path))
