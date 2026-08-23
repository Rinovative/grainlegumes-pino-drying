# ruff: noqa: ANN001, ANN003, ANN202, D100, D103, EM101, PLR2004, S101, SLF001, TRY003
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest
import torch

from src import datasets, domain
from src.analysis.eda import eda_dataframe as dataframe
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_workspace as workspace

if TYPE_CHECKING:
    from src.generation.cases.generation_cases_config import GenerationConfig
    from src.generation.runtime.generation_runtime_batch import TerminalBatchEvidence, TerminalCaseEvidence

_GIT_COMMIT = "b" * 40


def _case(index: int, *, digest: str = "a") -> SimpleNamespace:
    return SimpleNamespace(
        case_index=index,
        case_id=f"case_{index:04d}",
        case_input_id=digest * 64,
        simulation_case_id=digest * 64,
        success_sha256=digest * 64,
        provenance_sha256=digest * 64,
        case_hdf5_sha256=digest * 64,
    )


def _config(
    indices: tuple[int, ...] = (1, 2, 3),
    *,
    profile_id: str = "transient_drying",
    family: str = "lentil",
    regime: str = "natural",
    batch_id: str = "batch-id",
) -> SimpleNamespace:
    batch_name = f"{profile_id}__{family}__{regime}"
    return SimpleNamespace(
        material_family=family,
        sampling_regime=regime,
        batch_name=batch_name,
        batch_id=batch_id,
        batch_storage_name=f"{batch_name}__digest",
        profile=SimpleNamespace(
            id=profile_id,
            available_learning_views=(profile_id,),
            airflow_source="reference",
        ),
        template_sha256="a" * 64,
        material_role="id_source",
        evaluation_regime="held_out_family_ood",
        scientific_values={
            "scientific_fixed_values": {},
            "campaign_purpose": "family_generalization",
            "material_role": "id_source",
            "evaluation_regime": "held_out_family_ood",
        },
        case_indices=indices,
        case_id=lambda index: f"case_{index:04d}",
    )


def _partial_batch(
    *,
    cases: tuple[SimpleNamespace, ...],
    failed_indices: tuple[int, ...] = (),
    incomplete_indices: tuple[int, ...] = (),
    invalid_indices: tuple[int, ...] = (),
    profile_id: str = "transient_drying",
    family: str = "lentil",
    regime: str = "natural",
    batch_id: str = "batch-id",
    run_id: str = "run-id",
    campaign_state: str = "completed_with_failures",
) -> sources.GeneratedOutputEDABatch:
    expected_indices = tuple(sorted({case.case_index for case in cases} | set(failed_indices) | set(incomplete_indices) | set(invalid_indices)))
    config = _config(
        expected_indices,
        profile_id=profile_id,
        family=family,
        regime=regime,
        batch_id=batch_id,
    )
    return sources.GeneratedOutputEDABatch(
        source_kind="partial",
        generation_root=Path("/storage/01_generation"),
        batch_id=config.batch_id,
        batch_name=config.batch_name,
        batch_storage_name=config.batch_storage_name,
        simulation_profile=config.profile.id,
        available_learning_views=config.profile.available_learning_views,
        airflow_source=config.profile.airflow_source,
        material_family=config.material_family,
        sampling_regime=config.sampling_regime,
        template_sha256=config.template_sha256,
        git_commit=_GIT_COMMIT,
        scientific_values=config.scientific_values,
        cases=cast("tuple[TerminalCaseEvidence, ...]", cases),
        failed_case_indices=failed_indices,
        incomplete_case_indices=incomplete_indices,
        invalid_case_indices=invalid_indices,
        campaign_sources=((run_id, campaign_state),),
        config=cast("GenerationConfig", config),
    )


def _terminal_evidence(
    *,
    cases: tuple[SimpleNamespace, ...] = (),
    family: str = "lentil",
    regime: str = "natural",
    batch_id: str = "batch-id",
    storage_name: str = "terminal-storage",
) -> SimpleNamespace:
    return SimpleNamespace(
        generation_root=Path("/storage/01_generation"),
        batch_id=batch_id,
        batch_name=f"transient_drying__{family}__{regime}",
        batch_storage_name=storage_name,
        simulation_profile="transient_drying",
        available_learning_views=("transient_drying",),
        airflow_source="reference",
        material_family=family,
        sampling_regime=regime,
        template_sha256="a" * 64,
        git_commit=_GIT_COMMIT,
        campaign_purpose="family_generalization",
        cases=cases,
        scientific_config_payload=lambda: {
            "scientific_fixed_values": {},
            "campaign_purpose": "family_generalization",
            "material_role": "id_source",
            "evaluation_regime": "held_out_family_ood",
        },
    )


def _terminal_batch(**kwargs) -> sources.GeneratedOutputEDABatch:
    return sources._terminal_batch(cast("TerminalBatchEvidence", _terminal_evidence(**kwargs)))


def _catalog(
    *batches: sources.GeneratedOutputEDABatch,
    issues: tuple[sources.GeneratedOutputEDAIssue, ...] = (),
    total_issue_count: int | None = None,
) -> sources.GeneratedOutputEDACatalog:
    ordered = tuple(sorted(batches, key=lambda batch: (batch.material_family, batch.sampling_regime, batch.batch_id)))
    return sources.GeneratedOutputEDACatalog(
        batches=ordered,
        issues=issues,
        discovered_batch_count=len({batch.batch_id for batch in ordered}),
        complete_batch_count=sum(batch.source_kind == "terminal" for batch in ordered),
        partial_batch_count=sum(batch.source_kind == "partial" and bool(batch.cases) for batch in ordered),
        total_issue_count=len(issues) if total_issue_count is None else total_issue_count,
    )


def _campaign(config: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(
        profile=SimpleNamespace(available_learning_views=config.profile.available_learning_views),
        batches=(config,),
    )


def test_terminal_discovery_keeps_exact_evidence_and_deterministic_order(monkeypatch, tmp_path) -> None:
    terminals = {
        "z-storage": _terminal_evidence(family="lentil", regime="natural", batch_id="z", storage_name="z-storage"),
        "a-storage": _terminal_evidence(family="chickpea", regime="natural", batch_id="a", storage_name="a-storage"),
        "m-storage": _terminal_evidence(family="lentil", regime="dense", batch_id="m", storage_name="m-storage"),
    }
    monkeypatch.setattr(sources, "_terminal_storage_names", lambda _root: tuple(terminals))
    monkeypatch.setattr(sources, "_run_ids", lambda _root: ())
    monkeypatch.setattr(sources.generation.runtime, "admit_terminal_batch", lambda name, **_kwargs: terminals[name])

    catalog = sources.discover_generated_output_eda_catalog(storage_root=tmp_path)

    assert catalog.complete_batch_count == 3
    assert tuple(batch.batch_id for batch in catalog.batches) == ("a", "m", "z")
    assert all(batch.interpreter_batch is terminals[batch.batch_storage_name] for batch in catalog.batches)


def test_partial_successes_isolate_failed_missing_and_corrupt_siblings(monkeypatch, tmp_path) -> None:
    config = _config((1, 2, 3, 4, 5))
    partial = {
        "campaign_state": "completed_with_failures",
        "successful_cases": [{"batch_name": config.batch_name, "case_index": index, "state": "successful"} for index in (1, 2, 3, 4)],
        "failed_cases": [{"batch_name": config.batch_name, "case_index": 5, "state": "failed"}],
    }
    monkeypatch.setattr(
        sources.generation.publication.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {"state": "completed_with_failures", "git_commit": _GIT_COMMIT},
    )
    monkeypatch.setattr(sources.generation.publication.campaign_evidence, "campaign_from_manifest", lambda _manifest, **_kwargs: _campaign(config))
    monkeypatch.setattr(sources.generation.campaign, "read_partial_campaign_diagnostic_evidence", lambda *_args, **_kwargs: partial)
    monkeypatch.setattr(
        sources.generation.runtime,
        "completed_case_is_valid",
        lambda *_args, **_kwargs: pytest.fail("partial evidence must use direct case admission"),
    )
    admissions: list[tuple[int, str | None, str | None]] = []

    def admit(_config, index, **kwargs):
        admissions.append((index, kwargs.get("git_commit"), kwargs.get("validation_depth")))
        if index == 2:
            raise FileNotFoundError("missing case publication")
        if index == 3:
            raise ValueError("same-size tampered canonical output hash mismatch")
        if index == 4:
            raise RuntimeError("case identity mismatch")
        return _case(index)

    monkeypatch.setattr(sources.generation.runtime, "admit_completed_case", admit)

    batches, issues = sources._admit_campaign("run-id", storage_root=tmp_path)

    batch = batches[0]
    assert batch.available_case_count == 1
    assert batch.failed_case_count == 1
    assert batch.incomplete_case_count == 0
    assert batch.invalid_case_count == 3
    assert batch.discovered_case_count == 5
    assert admissions == [
        (1, _GIT_COMMIT, "full"),
        (2, _GIT_COMMIT, "full"),
        (3, _GIT_COMMIT, "full"),
        (4, _GIT_COMMIT, "full"),
    ]
    assert {issue.source_id for issue in issues} == {
        "run-id:case_0002",
        "run-id:case_0003",
        "run-id:case_0004",
    }
    assert any("tampered canonical output hash mismatch" in issue.message for issue in issues)
    assert any("case identity mismatch" in issue.message for issue in issues)


def test_active_campaign_classifies_completed_failed_incomplete_and_invalid_cases(monkeypatch, tmp_path) -> None:
    config = _config((1, 2, 3, 4))
    monkeypatch.setattr(
        sources.generation.publication.campaign_evidence,
        "load_campaign_run",
        lambda *_args, **_kwargs: {"state": "active", "git_commit": _GIT_COMMIT},
    )
    monkeypatch.setattr(sources.generation.publication.campaign_evidence, "campaign_from_manifest", lambda _manifest, **_kwargs: _campaign(config))
    monkeypatch.setattr(sources.generation.campaign, "read_partial_campaign_diagnostic_evidence", lambda *_args, **_kwargs: None)
    validity_calls: list[tuple[int, str | None]] = []
    failure_calls: list[tuple[int, str | None, str | None]] = []

    def completed(_config, index, **kwargs):
        validity_calls.append((index, kwargs.get("git_commit")))
        if index == 4:
            raise RuntimeError("corrupt completion")
        return index == 1

    def failed(_config, index, **kwargs):
        failure_calls.append((index, kwargs.get("execution_run_id"), kwargs.get("git_commit")))
        return index == 2

    monkeypatch.setattr(sources.generation.runtime, "completed_case_is_valid", completed)
    monkeypatch.setattr(sources.generation.runtime, "case_failure_is_recorded", failed)
    monkeypatch.setattr(sources.generation.runtime, "admit_completed_case", lambda _config, index, **_kwargs: _case(index))

    batches, issues = sources._admit_campaign("run-id", storage_root=tmp_path)

    batch = batches[0]
    assert tuple(case.case_index for case in batch.cases) == (1,)
    assert (batch.failed_case_count, batch.incomplete_case_count, batch.invalid_case_count) == (1, 1, 1)
    assert batch.discovered_case_count == 4
    assert validity_calls == [(1, _GIT_COMMIT), (2, _GIT_COMMIT), (3, _GIT_COMMIT), (4, _GIT_COMMIT)]
    assert failure_calls == [(2, "run-id", _GIT_COMMIT), (3, "run-id", _GIT_COMMIT)]
    assert len(issues) == 1
    assert "corrupt completion" in issues[0].message


def test_duplicate_sources_retain_one_compatible_most_complete_case_view() -> None:
    batches: dict[str, sources.GeneratedOutputEDABatch] = {}
    first = _partial_batch(cases=(_case(1),), incomplete_indices=(2,))
    duplicate = _partial_batch(cases=(_case(1),), incomplete_indices=(2,))
    superset = _partial_batch(cases=(_case(1), _case(2)))
    conflict = _partial_batch(cases=(_case(1, digest="c"), _case(2)))

    assert sources._merge_batch(batches, first) is None
    assert sources._merge_batch(batches, duplicate) is None
    assert sources._merge_batch(batches, superset) is None
    assert tuple(case.case_index for case in batches["batch-id"].cases) == (1, 2)
    assert sources._merge_batch(batches, conflict) is not None
    assert tuple(batches) == ("batch-id",)


def test_compatible_partial_runs_union_disjoint_and_overlapping_valid_cases() -> None:
    batches: dict[str, sources.GeneratedOutputEDABatch] = {}
    first = _partial_batch(
        cases=(_case(1),),
        incomplete_indices=(2, 3),
        run_id="run-a",
        campaign_state="active",
    )
    disjoint = _partial_batch(
        cases=(_case(2),),
        incomplete_indices=(1, 3),
        run_id="run-b",
        campaign_state="completed_with_failures",
    )
    overlapping = _partial_batch(
        cases=(_case(2), _case(3)),
        incomplete_indices=(1,),
        run_id="run-c",
        campaign_state="completed_with_failures",
    )

    assert sources._merge_batch(batches, first) is None
    assert sources._merge_batch(batches, disjoint) is None
    assert sources._merge_batch(batches, overlapping) is None

    merged = batches["batch-id"]
    assert tuple(case.case_index for case in merged.cases) == (1, 2, 3)
    assert merged.discovered_case_count == 3
    assert (merged.failed_case_count, merged.incomplete_case_count, merged.invalid_case_count) == (0, 0, 0)
    assert merged.campaign_run_id is None
    assert merged.campaign_state is None
    assert merged.campaign_sources == (
        ("run-a", "active"),
        ("run-b", "completed_with_failures"),
        ("run-c", "completed_with_failures"),
    )


def test_partial_merge_preserves_strongest_nonvalid_case_evidence() -> None:
    batches: dict[str, sources.GeneratedOutputEDABatch] = {}
    first = _partial_batch(
        cases=(_case(1),),
        failed_indices=(3,),
        incomplete_indices=(4,),
        invalid_indices=(2,),
        run_id="run-a",
        campaign_state="active",
    )
    second = _partial_batch(
        cases=(_case(1),),
        failed_indices=(4,),
        incomplete_indices=(2, 3),
        run_id="run-b",
        campaign_state="completed_with_failures",
    )

    assert sources._merge_batch(batches, first) is None
    assert sources._merge_batch(batches, second) is None

    merged = batches["batch-id"]
    assert merged.invalid_case_indices == (2,)
    assert merged.failed_case_indices == (3, 4)
    assert merged.incomplete_case_indices == ()
    assert merged.discovered_case_count == 4


def test_workspace_builds_one_unified_task_qualified_label_catalog(monkeypatch, tmp_path) -> None:
    steady = _partial_batch(
        cases=(_case(1),),
        profile_id="steady_flow",
        family="airflow_reference",
        regime="parameter_ood",
        batch_id="steady-id",
        run_id="steady-run",
    )
    transient_batch = _partial_batch(
        cases=(_case(1),),
        profile_id="transient_drying",
        family="lentil",
        regime="natural",
        batch_id="transient-id",
        run_id="transient-run",
    )
    monkeypatch.setattr(
        sources,
        "discover_generated_output_eda_catalog",
        lambda **_kwargs: _catalog(steady, transient_batch),
    )
    panel_inputs: list[tuple[object, object]] = []
    monkeypatch.setattr(
        workspace.panel,
        "build_eda_panel",
        lambda *, catalog, selection_state, **_kwargs: panel_inputs.append((catalog, selection_state)) or object(),
    )

    result = workspace.prepare_generated_output_eda_workspace(storage_root=tmp_path)

    assert len(result.catalog.views) == 2
    assert {view.simulation_profile for view in result.catalog.views} == {
        "steady_flow",
        "transient_drying",
    }
    labels = tuple(view.label for view in result.catalog.views)
    assert labels == (
        "Airflow · Airflow reference · fg · F OOD",
        "Drying · Lentil · fg · F OOD",
    )
    assert all("natural" not in label.casefold() for label in labels)
    assert all(label.endswith("F OOD") for label in labels)
    assert all(view.batch.batch_id not in view.label for view in result.catalog.views)
    assert all(not view.is_loaded for view in result.catalog.views)
    assert panel_inputs == [(result.catalog, result.selection_state)]


def test_notebook_and_workspace_public_api_have_no_task_selection() -> None:
    parameters = inspect.signature(workspace.prepare_generated_output_eda_workspace).parameters
    assert "task_id" not in parameters
    assert "task_ids" not in parameters
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "eda.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join(line for cell in notebook["cells"] for line in cell.get("source", ()))
    assert "TASK_ID" not in source
    assert "task_ids=" not in source


def test_terminal_workspace_uses_the_exact_strict_loader(monkeypatch, tmp_path) -> None:
    batch = _terminal_batch(cases=(_case(1),), family="lentil", regime="natural")
    frame = pd.DataFrame({"value": [1.0]}, index=pd.Index(["case_0001"], name="sample_id"))
    frame.attrs["task_id"] = "transient_drying"
    strict_calls: list[tuple[str, str, Path, int | None]] = []
    panel_inputs: list[tuple[object, object]] = []
    monkeypatch.setattr(sources, "discover_generated_output_eda_catalog", lambda **_kwargs: _catalog(batch))

    def strict(batch_name, *, task, storage_root, show_progress, max_cases):
        assert show_progress is False
        strict_calls.append((batch_name, task.id, storage_root, max_cases))
        return frame, []

    monkeypatch.setattr(workspace.dataframe, "generate_eda_dataframe", strict)
    monkeypatch.setattr(
        workspace.dataframe,
        "generate_eda_dataframe_from_completed_cases",
        lambda *_args, **_kwargs: pytest.fail("terminal evidence must stay on the strict loader"),
    )
    monkeypatch.setattr(
        workspace.panel,
        "build_eda_panel",
        lambda *, catalog, selection_state, **_kwargs: panel_inputs.append((catalog, selection_state)) or object(),
    )

    result = workspace.prepare_generated_output_eda_workspace(
        storage_root=tmp_path,
        max_cases=2,
    )

    assert result.panel is not None
    assert strict_calls == []
    assert len(panel_inputs) == 1
    assert panel_inputs[0] == (result.catalog, result.selection_state)
    assert result.catalog.views[0].is_loaded is False
    assert result.catalog.views[0].load() is frame
    assert strict_calls == [(batch.batch_storage_name, "transient_drying", tmp_path.resolve(), 2)]
    label = result.catalog.views[0].label
    assert label
    assert batch.batch_id not in label
    assert batch.batch_storage_name not in label


@pytest.mark.parametrize(("maximum", "expected_available"), [(None, 3), (1, 2)])
def test_partial_workspace_applies_the_bound_per_batch(monkeypatch, tmp_path, maximum, expected_available) -> None:
    batches = (
        _partial_batch(cases=(_case(1), _case(2)), family="lentil", batch_id="lentil-id"),
        _partial_batch(cases=(_case(1),), family="chickpea", batch_id="chickpea-id"),
    )
    calls: list[tuple[str, int | None]] = []
    monkeypatch.setattr(sources, "discover_generated_output_eda_catalog", lambda **_kwargs: _catalog(*batches))

    def materialize(batch, **kwargs):
        calls.append((batch.batch_id, kwargs["max_cases"]))
        selected = batch.cases if kwargs["max_cases"] is None else batch.cases[: kwargs["max_cases"]]
        frame = pd.DataFrame(
            {"value": [float(case.case_index) for case in selected]},
            index=pd.Index([case.case_id for case in selected], name="sample_id"),
        )
        frame.attrs["task_id"] = "transient_drying"
        return frame, []

    monkeypatch.setattr(workspace.dataframe, "generate_eda_dataframe_from_completed_cases", materialize)
    monkeypatch.setattr(workspace.panel, "build_eda_panel", lambda **_kwargs: object())

    result = workspace.prepare_generated_output_eda_workspace(
        storage_root=tmp_path,
        max_cases=maximum,
    )

    assert result.panel is not None
    assert calls == []
    assert sum(view.case_count for view in result.catalog.views) == expected_available
    assert all(not view.is_loaded for view in result.catalog.views)
    for view in result.catalog.views:
        view.load()
    assert calls == [("chickpea-id", maximum), ("lentil-id", maximum)]
    assert "Scientific payloads materialized during preparation: 0" in result.summary_text
    assert "Admitted completed source cases: 3" in result.summary_text


def test_workspace_reports_bounded_issues_and_actionable_empty_state(monkeypatch, tmp_path) -> None:
    issues = tuple(sources.GeneratedOutputEDAIssue(source_id=f"source-{index}", message="invalid") for index in range(5))
    catalog = _catalog(
        _partial_batch(
            cases=(),
            failed_indices=(1,),
            incomplete_indices=(2,),
            invalid_indices=(3,),
        ),
        issues=issues,
        total_issue_count=8,
    )
    monkeypatch.setattr(sources, "discover_generated_output_eda_catalog", lambda **_kwargs: catalog)

    result = workspace.prepare_generated_output_eda_workspace(
        storage_root=tmp_path,
        max_cases=None,
    )

    assert result.panel is None
    assert "Discovery issues: 8" in result.summary_text
    assert "3 additional source issues omitted" in result.summary_text
    assert "Failed cases: 1" in result.summary_text
    assert "Incomplete or running cases: 1" in result.summary_text
    assert "Invalid or corrupt cases: 1" in result.summary_text
    assert "No individually admitted completed cases" in result.summary_text


@pytest.mark.parametrize(
    ("maximum", "expected_count", "bounded_exclusions"),
    [(None, 3, {}), (2, 2, {"bounded_prefix": 1})],
)
def test_partial_steady_dataframe_preserves_semantics_and_complete_accounting(
    monkeypatch,
    maximum,
    expected_count,
    bounded_exclusions,
) -> None:
    task = domain.tasks.registry.get_task("steady_flow")
    batch = _partial_batch(
        cases=(_case(1), _case(2), _case(3)),
        failed_indices=(4,),
        incomplete_indices=(5,),
        invalid_indices=(6,),
        profile_id="steady_flow",
    )

    def interpret(_batch, case, *, task):
        inputs = torch.full((len(task.input_names), 2, 2), float(case.case_index))
        outputs = torch.full((len(task.output_names), 2, 2), float(case.case_index + 10))
        return (2, 2), inputs, outputs, {"case_id": case.case_id}, {}, "fingerprint"

    monkeypatch.setattr(datasets.packages.generated_batch, "interpret_generated_case", interpret)

    frame, _logs = dataframe.generate_eda_dataframe_from_completed_cases(
        batch,
        task=task,
        show_progress=False,
        max_cases=maximum,
    )

    assert len(frame) == expected_count
    assert tuple(frame.index) == tuple(f"case_{index:04d}" for index in range(1, expected_count + 1))
    assert "U" in frame
    assert frame.attrs["loaded_case_count"] == expected_count
    assert frame.attrs["available_case_count"] == 3
    assert frame.attrs["total_discovered_case_count"] == 6
    assert frame.attrs["failed_case_count"] == 1
    assert frame.attrs["incomplete_case_count"] == 1
    assert frame.attrs["invalid_case_count"] == 1
    assert frame.attrs["exclusion_reasons"] == {"invalid_or_corrupt": 1} | bounded_exclusions
    assert frame.attrs["case_accounting_scope"] == "individually_admitted_campaign_cases"


def test_strict_dataset_loader_still_requires_terminal_admission(monkeypatch) -> None:
    monkeypatch.setattr(
        datasets.packages.generated_batch.generation.runtime,
        "admit_terminal_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("terminal required")),
    )

    with pytest.raises(RuntimeError, match="terminal required"):
        datasets.packages.generated_batch.load_generated_batch("partial-batch", task_id="steady_flow")
