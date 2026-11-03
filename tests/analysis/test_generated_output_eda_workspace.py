"""Protect lazy generated-output EDA selection and shared workspace state.

Responsibilities:
  - Check catalog accounting, lazy source admission, and steady/transient field projection

Design principles:
  - Failed and pending candidates stay visible without fabricated success

This module does NOT:
  - Generate source campaigns or assign experiment task roles
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from support.generation_fixtures import successful_source_case

from src import common, datasets, generation
from src.analysis.eda import eda_dataframe as dataframe
from src.analysis.eda import eda_selection as selection
from src.analysis.eda import eda_sources as sources
from src.analysis.eda import eda_workspace as workspace
from src.analysis.eda.plots import eda_plot_transient as transient_plots

if TYPE_CHECKING:
    from pathlib import Path


def _row(
    case_id: str,
    *,
    outcome: str,
    selected: bool = False,
    domain_profile: str = "reference_range",
) -> dict:
    return {
        "task": "transient_drying",
        "campaign_name": "campaign-1",
        "source_name": "campaign-1__" + "a" * 16,
        "campaign_digest": "a" * 64,
        "material_family": "lentil",
        "domain_profile": domain_profile,
        "sampling_regime": "natural",
        "case_id": case_id,
        "case_input_id": "1" * 64 if outcome == "successful" else None,
        "simulation_case_id": "2" * 64 if outcome == "successful" else None,
        "case_hdf5_sha256": "3" * 64 if outcome == "successful" else None,
        "origin": "primary",
        "candidate_position": int(case_id[-1]),
        "outcome": outcome,
        "selected": selected,
        "attempts": (),
        "failure": None,
    }


def _campaign() -> dict:
    return {
        "campaign_name": "campaign-1",
        "source_name": "campaign-1__" + "a" * 16,
        "campaign_digest": "a" * 64,
        "campaign_purpose": "source_generation",
        "domain_profile": "reference_range",
        "original_design_sizes": {"lentil": 2},
    }


def test_catalog_preserves_current_campaign_decisions_without_opening_arrays(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = (
        _row("case_0001", outcome="successful", selected=True),
        _row("case_0002", outcome="failed"),
        _row("case_0003", outcome="pending"),
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaigns",
        lambda task, **_kwargs: (_campaign(),) if task == "transient_drying" else (),
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaign_cases",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "resolve_raw_case_reference",
        lambda *_args, **_kwargs: pytest.fail("preparation must keep scientific reads lazy"),
    )
    catalog = sources.discover_generated_output_eda_catalog(storage_root=tmp_path)
    assert catalog.discovered_campaign_count == 1
    assert catalog.discovered_candidate_count == 3
    assert len(catalog.batches) == 1
    batch = catalog.batches[0]
    assert (
        batch.available_case_count,
        batch.selected_case_count,
        batch.failed_case_count,
        batch.incomplete_case_count,
    ) == (1, 1, 1, 1)
    assert batch.cases[0].selector == {
        "campaign_name": "campaign-1",
        "campaign_digest": "a" * 64,
        "material_family": "lentil",
        "case_id": "case_0001",
        "case_input_id": "1" * 64,
        "simulation_case_id": "2" * 64,
        "case_hdf5_sha256": "3" * 64,
    }


def test_workspace_uses_role_neutral_labels_and_lazy_exact_source_admission(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = (_row("case_0001", outcome="successful", selected=True),)
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaigns",
        lambda task, **_kwargs: (_campaign(),) if task == "transient_drying" else (),
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaign_cases",
        lambda *_args, **_kwargs: rows,
    )
    calls = []
    admitted = SimpleNamespace(
        source_identity={"case_id": "case_0001"},
        design_provenance={"origin": "primary"},
    )

    def resolve(task: str, selector: Any, **_kwargs: Any) -> Any:
        calls.append((task, selector))
        return admitted

    monkeypatch.setattr(sources.generation.publication.catalog, "resolve_raw_case_reference", resolve)
    prepared = workspace.prepare_generated_output_eda_workspace(storage_root=tmp_path, max_cases=1)
    assert calls == []
    assert "campaign=campaign-1" in prepared.summary_text
    assert "material_role" not in prepared.summary_text
    view = prepared.catalog.views[0]
    assert isinstance(view, selection.GeneratedOutputEDAView)
    assert not view.is_loaded
    assert view.case_ids == ("case_0001",)
    assert view.batch.admit_case(view.batch.cases[0]) is admitted
    assert calls == [("transient_drying", view.batch.cases[0].selector)]
    assert "Campaign: campaign-1" in view.tooltip


def test_campaign_accounting_and_empty_state_are_truthful(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sources,
        "discover_generated_output_eda_catalog",
        lambda **_kwargs: sources.GeneratedOutputEDACatalog(
            batches=(),
            issues=(sources.GeneratedOutputEDAIssue("bad-campaign", "invalid evidence"),),
            discovered_campaign_count=1,
            discovered_candidate_count=2,
            total_issue_count=1,
        ),
    )
    prepared = workspace.prepare_generated_output_eda_workspace(storage_root=tmp_path)
    assert prepared.panel is None
    assert "Discovered campaigns: 1" in prepared.summary_text
    assert "Discovered candidates: 2" in prepared.summary_text
    assert "invalid evidence" in prepared.summary_text
    assert "No individually admitted completed cases" in prepared.summary_text


def test_failed_and_pending_source_candidates_remain_visible_without_successes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = (
        _row("case_0001", outcome="failed"),
        _row("case_0002", outcome="pending"),
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaigns",
        lambda task, **_kwargs: (_campaign(),) if task == "transient_drying" else (),
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaign_cases",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "resolve_raw_case_reference",
        lambda *_args, **_kwargs: pytest.fail("failed or pending candidates have no loadable scientific payload"),
    )

    prepared = workspace.prepare_generated_output_eda_workspace(storage_root=tmp_path)

    assert prepared.panel is None
    assert prepared.catalog.views == ()
    source_catalog = prepared.catalog.source_catalog
    assert isinstance(source_catalog, sources.GeneratedOutputEDACatalog)
    assert len(source_catalog.batches) == 1
    batch = source_catalog.batches[0]
    assert batch.cases == ()
    assert [row["outcome"] for row in batch.candidate_rows] == ["failed", "pending"]
    assert (batch.failed_case_count, batch.incomplete_case_count) == (1, 1)
    assert "Failed cases: 1" in prepared.summary_text
    assert "Incomplete or running cases: 1" in prepared.summary_text
    assert "0 successful, 0 selected, 1 failed, 1 incomplete" in prepared.summary_text


def test_published_transient_eda_uses_actual_inputs_targets_and_bounded_cases(transient_package: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    storage = transient_package["root"]
    dataset_id = transient_package["train_dataset"]
    calls = []
    original = datasets.runtime.factory.create_dataset

    def load(request: Any, **kwargs: Any) -> Any:
        calls.append(request)
        return original(request, **kwargs)

    monkeypatch.setattr(datasets.runtime.factory, "create_dataset", load)
    prepared = workspace.prepare_published_dataset_eda_workspace(storage_root=storage, max_cases=1)
    assert calls == []
    view = prepared.catalog.view(dataset_id)
    assert isinstance(view, selection.PublishedDatasetEDAView)
    assert len(view.case_ids) == 1
    assert not view.is_loaded
    frame = view.load()
    assert view.load() is frame
    assert len(calls) == 1
    assert frame.attrs["available_case_count"] == 4
    assert frame.attrs["loaded_case_count"] == 1
    values = {(value.group, value.name): value for value in dataframe.published_tensor_values(frame, view.case_ids[0], transition=0)}
    assert {group for group, _name in values} == {
        "state",
        "static",
        "boundary",
        "scalars",
        "target",
    }
    state_trajectories = frame.loc[view.case_ids[0], "state_trajectories"]
    assert isinstance(state_trajectories, dict)
    states = cast("dict[str, np.ndarray]", state_trajectories)["T"]
    np.testing.assert_allclose(values["target", "delta_T"].array, states[1] - states[0], rtol=0.0, atol=2.0e-5)
    assert values["target", "delta_T"].unit == "K"
    static_fields = frame.loc[view.case_ids[0], "static_fields"]
    assert isinstance(static_fields, dict)
    np.testing.assert_array_equal(
        values["static", "rho_bu_dry"].array,
        cast("dict[str, np.ndarray]", static_fields)["rho_bu_dry"],
    )
    assert values["static", "rho_bu_dry"].array.shape == (16, 16)
    assert all(value.array.ndim == 0 for (group, _), value in values.items() if group in {"scalars", "boundary"})
    assert frame.attrs["field_categories"]["dynamic_state"] == (
        "T",
        "phi",
        "w_surf",
        "w_int",
    )
    assert "schedule" not in view.capabilities
    figures = (
        transient_plots.plot_state_snapshot_comparison(
            datasets={"Published": frame},
            case_ids={"Published": view.case_ids[0]},
            physical_time_hours=1.0,
            channels=("T",),
        ),
        transient_plots.plot_state_trajectory_summary(datasets={"Published": frame}, max_cases=1, channels=("T",)),
    )
    for figure in figures:
        assert figure.axes
        plt.close(figure)
    time = frame.loc[view.case_ids[0], "time"]
    assert isinstance(time, dict)
    transition_count = len(cast("dict[str, np.ndarray]", time)["transition_t_n_hours"])
    with pytest.raises(ValueError, match="exact stored transition"):
        dataframe.published_tensor_values(frame, view.case_ids[0], transition=transition_count)


@pytest.mark.usefixtures("_dataset_producer_evidence")
def test_published_steady_eda_preserves_encoded_inputs_and_boundary_support(
    generation_config_factory: Any,
    fake_comsol: Any,
    tmp_path: Path,
) -> None:
    config_path, _template = generation_config_factory(
        simulation_profile="steady_flow",
        executable=fake_comsol,
        natural_count=1,
        retain_raw_csv=False,
        retain_solved_model=False,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    batch = campaign.batches[0]
    storage = tmp_path / "published-steady"
    generation.lifecycle.create_campaign(campaign, storage_root=storage)
    inputs = generation.cases.input_generation.generate_input_cases(batch, 1, storage_root=storage)
    successful_source_case(campaign.source_name, batch, 1, storage, tmp_path / "work")
    run_id = generation.campaign.campaign_run_id(campaign, git_commit="a" * 40)
    run_dir = generation.publication.campaign_evidence.campaign_run_directory(run_id, storage_root=storage)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = generation.campaign._new_campaign_manifest(
        campaign,
        run_id=run_id,
        requested_commit="a" * 40,
        run_directory=run_dir,
        scheduler_log_directory=tmp_path / "logs",
        storage_root=storage,
        input_sources={
            batch.batch_name: {
                "input_generation_id": inputs.input_generation_id,
                "source_git_commit": "a" * 40,
                "case_indices": list(batch.case_indices),
            }
        },
    )
    common.serialization.atomic_write_json(
        generation.publication.campaign_evidence.campaign_run_manifest_path(run_id, storage_root=storage), manifest
    )
    row = generation.publication.catalog.list_campaign_cases("steady_flow", campaign.source_name, storage_root=storage)[0]
    selector = {
        key: row[key]
        for key in (
            "campaign_name",
            "campaign_digest",
            "material_family",
            "case_id",
            "case_input_id",
            "simulation_case_id",
            "case_hdf5_sha256",
        )
    }
    result = datasets.packages.build_dataset_selection(
        {
            "schema_kind": "dataset_selection_request",
            "schema_version": 2,
            "source_task": "steady_flow",
            "dataset_name": "steady_eda_source",
            "dataset_view": "steady_flow",
            "selection": {"kind": "exact"},
            "source_cases": [selector],
        },
        storage_root=storage,
    )
    prepared = workspace.prepare_published_dataset_eda_workspace(storage_root=storage)
    view = prepared.catalog.view(result["dataset_id"])
    frame = view.load()
    values = {value.name: value for value in dataframe.published_tensor_values(frame, view.case_ids[0])}
    assert values["Kxx"].unit == "1"
    assert values["Kxx"].representation == "dimensionless_log10_ratio_to_1_m2"
    assert frame.attrs["field_units"]["Kxx"] == "1"
    assert frame.attrs["field_units"]["Kxy"] == "1"
    np.testing.assert_array_equal(values["Kxx"].array, frame.iloc[0]["Kxx"])
    assert values["p"].purpose == "target"
    assert values["p"].unit == "Pa"
    assert values["p_in_bc"].array.ndim == 1
    assert values["p_in_bc"].support == "inlet boundary"
    figure = transient_plots.plot_spatial_field_comparison(
        datasets={"Published": frame},
        case_ids={"Published": view.case_ids[0]},
        fields=("Kxx", "Kxy", "u"),
    )
    plt.close(figure)


def test_output_workspace_uses_normal_campaign_catalog_and_case_accounting(
    generation_config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose normal campaign outcomes with complete case accounting."""
    config_path, _template = generation_config_factory(
        simulation_profile="transient_drying",
        natural_count=1,
    )
    campaign = generation.cases.config.load_campaign_config(config_path)
    generation.lifecycle.create_campaign(campaign, storage_root=tmp_path)
    rows = tuple(
        _row(f"case_{index:04d}", outcome=outcome, selected=outcome == "successful")
        | {"campaign_digest": campaign.campaign_digest, "source_name": campaign.source_name}
        for index, outcome in enumerate(("successful", "failed", "pending"), start=1)
    )
    monkeypatch.setattr(
        sources.generation.publication.catalog,
        "list_campaign_cases",
        lambda *_args, **_kwargs: rows,
    )
    monkeypatch.setattr(workspace.panel, "build_eda_panel", lambda **_kwargs: None)
    included = workspace.prepare_generated_output_eda_workspace(storage_root=tmp_path)
    assert len(included.catalog.views) == 1
    view = included.catalog.views[0]
    assert isinstance(view, selection.GeneratedOutputEDAView)
    batch = view.batch
    assert (
        batch.available_case_count,
        batch.failed_case_count,
        batch.incomplete_case_count,
    ) == (1, 1, 1)


def test_source_versions_with_one_design_name_remain_separately_selectable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish changed Source content without changing its scientific design label."""
    campaigns = tuple({**_campaign(), "campaign_digest": digit * 64, "source_name": "campaign-1__" + digit * 16} for digit in ("a", "b"))
    by_source = {
        campaign["source_name"]: (
            {**_row("case_0001", outcome="successful"), "campaign_digest": campaign["campaign_digest"], "source_name": campaign["source_name"]},
        )
        for campaign in campaigns
    }
    monkeypatch.setattr(
        sources.generation.publication.catalog, "list_campaigns", lambda task, **_kwargs: campaigns if task == "transient_drying" else ()
    )
    monkeypatch.setattr(sources.generation.publication.catalog, "list_campaign_cases", lambda _task, source, **_kwargs: by_source[source])
    catalog = sources.discover_generated_output_eda_catalog(storage_root=tmp_path)
    assert not catalog.issues
    assert {batch.campaign_name for batch in catalog.batches} == {"campaign-1"}
    assert {batch.source_name for batch in catalog.batches} == set(by_source)
    assert len({batch.batch_id for batch in catalog.batches}) == len(campaigns)
