"""Protect the (material, role) dataset path of steady, comparison, and transient evaluation.

Responsibilities:
  - Check that a cut frame recomputes the material-balanced objective of the artifact contract, per material and pooled
  - Check role and material admission: unavailable materials, refused roles, and case-matched comparison cuts
  - Check that transient inventories take the declared case role and resolve choices to frames and cases

Design principles:
  - The expected objective is computed by hand from per-case errors, so a row mean or a mean of roots fails
  - Frames are small synthetic steady tables built from the real case statistics and metric columns

This module does NOT:
  - Load artifacts from disk or draw any view
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import torch
from support.steady_evaluation import steady_frame

from src import analysis, domain, learning
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as datasets
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.evaluation import evaluation_session
from src.analysis.evaluation import evaluation_transient_session as transient_session
from src.analysis.presentation import dataset_selection as selection

if TYPE_CHECKING:
    from collections.abc import Sequence

_TASK = domain.tasks.steady_flow.STEADY_FLOW
_SCALES = dict.fromkeys(_TASK.output_names, 1.0)
_FLOORS = {field: _TASK.reference_floor(field) for field in _TASK.output_names}
_SPACING = (0.25, 0.5)
_REFERENCE = np.linspace(1.0, 2.0, len(_TASK.output_names) * 6 * 7).reshape(len(_TASK.output_names), 6, 7)
_PRESSURE_WEIGHT = 0.5  # Two output groups weigh equally; the velocity group has no error here.


def _source_index(case_id: str) -> int:
    """Give every case id one fixed source index, so models that share a case share its saved source membership."""
    return int.from_bytes(case_id.encode(), "big")


def _frame(cases: Sequence[tuple[str, str, float]], *, split_role: str = "eval") -> pd.DataFrame:
    """Build one steady role frame; every case has a constant pressure error and an exact velocity."""
    contracts = analysis.artifacts.contracts
    rows: list[dict[str, object]] = []
    for case_id, material, error in cases:
        prediction = _REFERENCE.copy()
        prediction[0] += error
        sums = learning.metrics.metrics.field_statistics(
            torch.from_numpy(prediction[None]),
            torch.from_numpy(_REFERENCE[None]),
            grid_spacing=_SPACING,
        )[0]
        row: dict[str, object] = {
            "simulation_case_id": case_id,
            "material_family": material,
            "source_index": _source_index(case_id),
            "prediction_validity": json.dumps(validity_contract.build_field_prediction_validity(prediction, channels=_TASK.output_names)),
        }
        for index, field in enumerate(_TASK.output_names):
            row.update(zip(contracts.case_statistic_columns(field), sums[index].tolist(), strict=True))
        row.update(
            contracts.case_metric_values(
                sums, fields=_TASK.output_names, output_groups=_TASK.output_groups, train_scales=_SCALES, reference_floors=_FLOORS
            )
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    groups = contracts.output_group_payload(_TASK.output_groups)
    aggregate = contracts.aggregate_normalized_group_macro_rmse(frame, output_groups=groups, train_scales=_SCALES, reference_floors=_FLOORS)
    selection = {
        "effective_ordered_source_indices_sha256": contracts.ordered_indices_sha256(frame["source_index"].tolist()),
        "effective_case_count": len(frame),
    }
    frame.attrs.update(
        {
            "provenance_complete": True,
            "artifact_provenance": {
                "evaluator": {"output_groups": groups},
                "split_role": split_role,
                "selection": selection,
                "producer": {"analysis_sha256": "0" * 64},
            },
            "train_scales": dict(_SCALES),
            "reference_floors": dict(_FLOORS),
            "dataset_role": split_role,
            dataframe.PRIMARY_OBJECTIVE_ID: aggregate,
        }
    )
    return frame


def _objective(frame: pd.DataFrame) -> float:
    return frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"]


def test_steady_inventory_reads_the_role_from_the_split_and_the_materials_from_the_rows() -> None:
    frames = {"id": _frame([("a", "lentil", 1.0), ("b", "chickpea", 1.0)]), "ood": _frame([("c", "lentil", 1.0)], split_role="parameter_ood")}

    inventory = datasets.steady_inventory(frames)

    assert inventory.poolable
    assert [(item.key, item.label) for item in inventory.items] == [
        (("chickpea", "id"), "chickpea, ID"),
        (("lentil", "id"), "lentil, ID"),
        (("lentil", "parameter_ood"), "lentil, Parameter OOD"),
    ]
    with pytest.raises(ValueError, match="one frame per role"):
        datasets.steady_inventory({"first": frames["id"], "second": frames["id"]})
    with pytest.raises(ValueError, match="explicit split role"):
        datasets.steady_inventory({"bare": pd.DataFrame({"simulation_case_id": ["a"], "material_family": ["lentil"]})})


def test_steady_cuts_union_to_the_rows_keep_run_attributes_and_leave_the_source_untouched() -> None:
    frame = _frame([("a", "lentil", 1.0), ("b", "chickpea", 3.0), ("c", "chickpea", 3.0), ("d", "kidney_bean", 2.0)])
    objective_before = _objective(frame)
    choices = selection.resolve_choices(
        datasets.steady_inventory({"id": frame}), [("lentil", "id"), ("chickpea", "id"), ("kidney_bean", "id")], pooled=False
    )

    cuts = datasets.steady_frames({"id": frame}, choices)

    assert list(cuts) == ["chickpea, ID", "kidney bean, ID", "lentil, ID"]
    assert sum(len(cut) for cut in cuts.values()) == len(frame)
    assert set(pd.concat(cuts.values())["simulation_case_id"]) == set(frame["simulation_case_id"])
    for cut in cuts.values():
        assert cut.attrs["source_artifact_provenance"] == frame.attrs["artifact_provenance"]
        assert cut.attrs["train_scales"] == frame.attrs["train_scales"]
    assert _objective(frame) == objective_before


def test_cut_objective_is_material_balanced_with_the_roots_after_the_means() -> None:
    # One lentil case with error 1 and three chickpea cases with error 3: the case-weighted mean square is 7,
    # the mean of the per-material roots is 2, and only the material-balanced reduction gives sqrt((1 + 9) / 2).
    frame = _frame([("l1", "lentil", 1.0), ("c1", "chickpea", 3.0), ("c2", "chickpea", 3.0), ("c3", "chickpea", 3.0)])
    inventory = datasets.steady_inventory({"id": frame})

    separate = datasets.steady_frames({"id": frame}, selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=False))
    pooled = datasets.steady_frames({"id": frame}, selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=True))

    assert _objective(separate["lentil, ID"]) == pytest.approx(_PRESSURE_WEIGHT * 1.0)
    assert _objective(separate["chickpea, ID"]) == pytest.approx(_PRESSURE_WEIGHT * 3.0)
    assert list(pooled) == ["ID, alle Materialien"]
    assert _objective(pooled["ID, alle Materialien"]) == pytest.approx(_PRESSURE_WEIGHT * math.sqrt(5.0))
    assert _objective(pooled["ID, alle Materialien"]) == pytest.approx(_objective(frame))
    assert _objective(pooled["ID, alle Materialien"]) != pytest.approx(_PRESSURE_WEIGHT * math.sqrt(7.0))
    assert _objective(pooled["ID, alle Materialien"]) != pytest.approx(_PRESSURE_WEIGHT * 2.0)


def test_steady_cut_of_an_unknown_role_or_material_or_an_unavailable_choice_is_refused() -> None:
    frame = _frame([("a", "lentil", 1.0)])
    inventory = datasets.steady_inventory(
        {"id": frame}, unavailable=[(selection.DatasetItem(("rapeseed", "id"), "rapeseed", "id", "rapeseed, ID"), "fehlt")]
    )
    unavailable = selection.resolve_choices(inventory, [("rapeseed", "id")], pooled=False)
    other_role = [selection.DatasetChoice("material_ood", (inventory.items[0],), True, "x")]
    other_material = [selection.DatasetChoice("id", (selection.DatasetItem(("rapeseed", "id"), "rapeseed", "id", "r"),), True, "y")]

    for choices, match in ((unavailable, "unavailable"), (other_role, "holds role"), (other_material, "holds no cases")):
        with pytest.raises(ValueError, match=match):
            datasets.steady_frames({"id": frame}, choices)


def test_comparison_inventory_keeps_unshared_materials_and_refused_roles_visible_with_reasons() -> None:
    run_a = _frame([("l1", "lentil", 1.0), ("c1", "chickpea", 1.0), ("p1", "field_pea", 1.0)])
    run_b = _frame([("l1", "lentil", 2.0), ("k1", "kidney_bean", 2.0), ("p2", "field_pea", 2.0)])

    inventory = datasets.comparison_inventory({"id": {"run a": run_a, "run b": run_b}}, refused={"parameter_ood": "Rolle ist nicht verfügbar"})

    assert [item.label for item in inventory.items] == ["lentil, ID"]
    reasons = {item.label: reason for item, reason in inventory.unavailable}
    assert reasons["chickpea, ID"] == "keine gemeinsamen Fälle in run b"
    assert reasons["kidney bean, ID"] == "keine gemeinsamen Fälle in run a"
    assert reasons["field pea, ID"] == "keine gemeinsamen Fälle in run a, run b"  # both hold it, but no case is shared
    assert reasons["Parameter OOD"] == "Rolle ist nicht verfügbar"
    with pytest.raises(ValueError, match="both refused and loaded"):
        datasets.comparison_inventory({"id": {"run a": run_a}}, refused={"id": "x"})
    with pytest.raises(ValueError, match="was given for role"):
        datasets.comparison_inventory({"material_ood": {"run a": run_a}}, refused={})


def test_comparison_cut_keeps_the_shared_cases_per_material_and_recomputes_each_objective() -> None:
    run_a = _frame([("l1", "lentil", 1.0), ("l2", "lentil", 3.0), ("c1", "chickpea", 5.0), ("c2", "chickpea", 7.0)])
    run_b = _frame([("l2", "lentil", 3.0), ("l3", "lentil", 9.0), ("c1", "chickpea", 5.0), ("c3", "chickpea", 1.0)])
    frames_by_model = {"run a": run_a, "run b": run_b}
    inventory = datasets.comparison_inventory({"id": frames_by_model}, refused={})
    pooled = selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=True)

    cuts = datasets.comparison_frames(frames_by_model, pooled[0])

    assert list(cuts) == ["run a", "run b"]
    for cut in cuts.values():
        assert sorted(cut["simulation_case_id"]) == ["c1", "l2"]
        assert _objective(cut) == pytest.approx(_PRESSURE_WEIGHT * math.sqrt((9.0 + 25.0) / 2))
    with pytest.raises(ValueError, match="at least one model"):
        datasets.comparison_frames({}, pooled[0])


def test_cuts_name_their_dataset_panel_and_model_series() -> None:
    run_a = _frame([("l1", "lentil", 1.0), ("c1", "chickpea", 5.0)])
    run_b = _frame([("l1", "lentil", 3.0), ("c1", "chickpea", 7.0)])
    inventory = datasets.comparison_inventory({"id": {"run a": run_a, "run b": run_b}}, refused={})
    choice = selection.resolve_choices(inventory, [("lentil", "id")], pooled=False)[0]

    compared = datasets.comparison_frames({"run a": run_a, "run b": run_b}, choice)
    single = datasets.steady_frames({"id": run_a}, [choice], model="run a")

    assert dict(compared.labels) == {"run a": (choice.label, "run a"), "run b": (choice.label, "run b")}
    assert dict(single.labels) == {choice.label: (choice.label, "run a")}
    assert all("display_model" not in cut.attrs for cut in compared.values())


def test_cuts_carry_their_effective_membership_to_the_comparison_admission_and_keep_the_source_provenance() -> None:
    run_a = _frame([("l1", "lentil", 1.0), ("l2", "lentil", 3.0), ("c1", "chickpea", 5.0)])
    run_b = _frame([("l2", "lentil", 3.0), ("l3", "lentil", 9.0), ("c1", "chickpea", 5.0)])
    frames_by_model = {"run a": run_a, "run b": run_b}
    inventory = datasets.comparison_inventory({"id": frames_by_model}, refused={})
    pooled = selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=True)
    with pytest.raises(dataframe.ComparisonCompatibilityError):  # the original memberships differ
        dataframe.validate_comparison(frames_by_model, verify_grid=False)

    cuts = datasets.comparison_frames(frames_by_model, pooled[0])

    dataframe.validate_comparison(cuts, verify_grid=False)
    for model, cut in cuts.items():
        assert cut.attrs["source_artifact_provenance"] == frames_by_model[model].attrs["artifact_provenance"]
        assert cut.attrs["artifact_provenance"]["selection"]["effective_case_count"] == len(cut) == 2
        assert cut.attrs["artifact_provenance"]["selection"]["material_families"] == ["chickpea", "lentil"]


def _two_material_frame(tmp_path: Path) -> pd.DataFrame:
    """Return the steady fixture frame with its second case moved to a second material."""
    frame, _grid = steady_frame(tmp_path)
    frame.loc[1, "material_family"] = "chickpea"
    return frame


def _separate_choices(frame: pd.DataFrame) -> tuple[selection.DatasetChoice, ...]:
    inventory = datasets.steady_inventory({"id": frame})
    return selection.resolve_choices(inventory, [item.key for item in inventory.items], pooled=False)


def test_separate_material_cuts_of_one_live_run_are_comparable_and_draw_from_the_live_session(tmp_path: Path) -> None:
    frame = _two_material_frame(tmp_path)
    with evaluation_session.EvaluationSession({"id": frame}) as live:
        cuts = datasets.steady_frames({"id": frame}, _separate_choices(frame))

        dataframe.validate_comparison(cuts)
        for cut in cuts.values():
            with evaluation_session.scoped_session({"cut": cut}) as scoped:
                assert scoped is live
                assert scoped.full_summary(cut).sample_count == len(cut) == 1
    run_a = _frame([("l1", "lentil", 1.0), ("c1", "chickpea", 1.0)])
    run_b = _frame([("l2", "lentil", 1.0), ("c1", "chickpea", 1.0)])
    lentil = {}
    for run, source in (("run a", run_a), ("run b", run_b)):
        choices = selection.resolve_choices(datasets.steady_inventory({"id": source}), [("lentil", "id")], pooled=False)
        lentil[run] = datasets.steady_frames({"id": source}, choices)["lentil, ID"]
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="membership"):  # one choice, different cases
        dataframe.validate_comparison(lentil, verify_grid=False)


def test_a_whole_frame_and_a_cut_of_one_role_are_refused_together() -> None:
    run_a = _frame([("l1", "lentil", 1.0), ("c1", "chickpea", 1.0)])
    run_b = _frame([("l1", "lentil", 2.0), ("c1", "chickpea", 2.0)])
    choices = selection.resolve_choices(datasets.steady_inventory({"id": run_b}), [("lentil", "id")], pooled=False)
    cut_b = datasets.steady_frames({"id": run_b}, choices)["lentil, ID"]

    with pytest.raises(dataframe.ComparisonCompatibilityError, match="'eval'"):
        dataframe.validate_comparison({"run a": run_a, "run b": cut_b}, verify_grid=False)


def test_one_session_keeps_one_canonical_frame_per_run_and_role(tmp_path: Path) -> None:
    frame = _two_material_frame(tmp_path)
    cuts = datasets.steady_frames({"id": frame}, _separate_choices(frame))

    with pytest.raises(ValueError, match="run-role key is already bound"):
        evaluation_session.EvaluationSession(cuts)
    assert all(evaluation_session.bound_session(cut) is None for cut in cuts.values())


def test_a_bound_cut_changed_after_binding_is_refused_on_the_next_draw(tmp_path: Path) -> None:
    frame = _two_material_frame(tmp_path)
    lentil = [choice for choice in _separate_choices(frame) if choice.label == "lentil, ID"]
    with evaluation_session.EvaluationSession({"id": frame}):
        cut = datasets.steady_frames({"id": frame}, lentil)["lentil, ID"]
        cut.attrs[dataframe.PRIMARY_OBJECTIVE_ID] = {**cut.attrs[dataframe.PRIMARY_OBJECTIVE_ID], "value": 0.0}

        with pytest.raises(evaluation_session.EvaluationArtifactChangedError, match="cut changed"):
            datasets.steady_frames({"id": frame}, lentil)


def test_a_session_whose_second_frame_fails_to_bind_releases_the_first(tmp_path: Path) -> None:
    frame, _grid = steady_frame(tmp_path)

    with pytest.raises(ValueError, match="already bound"):
        evaluation_session.EvaluationSession({"first": frame, "same run and cases": frame.copy()})
    assert evaluation_session.bound_session(frame) is None


def test_a_cut_shares_no_nested_attribute_object_with_the_source_frame() -> None:
    frame = _frame([("a", "lentil", 1.0), ("b", "chickpea", 3.0)])
    frame.attrs["_exact_comparison_grid_cache"] = ("witnesses of the whole frame",)
    frame.attrs["artifact_provenance"]["selection"]["nested"] = {"note": ["kept"]}
    snapshot = copy.deepcopy(frame.attrs)
    choices = selection.resolve_choices(datasets.steady_inventory({"id": frame}), [("lentil", "id")], pooled=False)

    cut = datasets.steady_frames({"id": frame}, choices)["lentil, ID"]
    cut.attrs["train_scales"].clear()
    cut.attrs["artifact_provenance"]["evaluator"]["output_groups"].clear()
    cut.attrs["artifact_provenance"]["selection"]["nested"]["note"].clear()

    assert frame.attrs == snapshot
    cut.attrs["source_artifact_provenance"]["evaluator"]["output_groups"].clear()
    assert frame.attrs == snapshot
    assert "_exact_comparison_grid_cache" not in cut.attrs  # it describes the rows of the whole frame, not the cut


def test_display_labels_live_per_call_and_leave_the_bound_frame_untouched(tmp_path: Path) -> None:
    frame = _two_material_frame(tmp_path)
    lentil = [choice for choice in _separate_choices(frame) if choice.label == "lentil, ID"]
    with evaluation_session.EvaluationSession({"id": frame}):
        first = datasets.steady_frames({"id": frame}, lentil, model="run a")
        attrs = copy.deepcopy(first["lentil, ID"].attrs)
        attrs.pop(evaluation_session.SESSION_ATTR)

        again = datasets.steady_frames({"id": frame}, lentil, model="run b")
        unnamed = datasets.steady_frames({"id": frame}, lentil)

        assert again["lentil, ID"] is first["lentil, ID"] is unnamed["lentil, ID"]
        assert [r.labels["lentil, ID"] for r in (first, again, unnamed)] == [("lentil, ID", m) for m in ("run a", "run b", "")]
        remaining = copy.deepcopy(first["lentil, ID"].attrs)
        remaining.pop(evaluation_session.SESSION_ATTR)
        assert remaining == attrs


def test_a_live_session_binds_each_cut_once_and_redraws_hit_its_caches(tmp_path: Path) -> None:
    frame = _two_material_frame(tmp_path)
    lentil = [choice for choice in _separate_choices(frame) if choice.label == "lentil, ID"]
    comparison = datasets.comparison_inventory({"id": {"run": frame}}, refused={})
    pooled = selection.resolve_choices(comparison, [item.key for item in comparison.items], pooled=True)[0]
    with evaluation_session.EvaluationSession({"id": frame}) as live:
        first = datasets.steady_frames({"id": frame}, lentil)["lentil, ID"]
        assert live.full_summary(first).sample_count == 1
        hits = live.counters["aggregate_hits"]

        second = datasets.steady_frames({"id": frame}, lentil)["lentil, ID"]

        assert second is first
        with evaluation_session.scoped_session({"cut": second}) as scoped:
            assert scoped is live
            assert scoped.full_summary(second).sample_count == 1
        assert live.counters["aggregate_hits"] == hits + 1
        every_material = datasets.comparison_frames({"run": frame}, pooled)["run"]  # same membership as its source, no collision
        assert evaluation_session.bound_session(every_material) is live
        assert every_material is not frame
        assert live.full_summary(every_material).sample_count == len(frame)
        assert evaluation_session.bound_session(frame) is live


def test_steady_frames_reject_two_frames_of_one_role() -> None:
    frame = _frame([("a", "lentil", 1.0)])
    choices = selection.resolve_choices(datasets.steady_inventory({"id": frame}), [("lentil", "id")], pooled=False)

    with pytest.raises(ValueError, match="one frame per role"):
        datasets.steady_frames({"first": frame, "second": _frame([("b", "lentil", 2.0)])}, choices)


def _entry(frame_name: str, case_id: str, material: str, artifact_role: str, assigned: str | None) -> transient_session.TransientCaseInventoryEntry:
    return transient_session.TransientCaseInventoryEntry(
        frame_name=frame_name,
        case_id=case_id,
        case_label=case_id,
        material_family=material,
        dataset_role=artifact_role,  # type: ignore[arg-type]
        dataset_name="dataset",
        source_dataset_ids=("dataset",),
        membership_digests=(),
        artifact_root=Path("artifact"),
        artifact_identity_sha256="0" * 64,
        dataset_identity={},
        evaluation_assignment=None if assigned is None else {"role": assigned},
    )


def test_transient_inventory_takes_the_declared_case_role_and_resolves_choices_to_frames_and_cases() -> None:
    entries = (
        _entry("ID", "c1", "lentil", "id", None),
        _entry("ID", "c2", "chickpea", "id", "id"),
        _entry("OOD", "c3", "lentil", "material_ood", "parameter_ood"),
        _entry("OOD", "c4", "lentil", "material_ood", "parameter_ood"),
    )
    session = SimpleNamespace(case_inventory=lambda: entries)

    inventory = datasets.transient_inventory(session, refused={"material_ood": "Rolle ist nicht verfügbar"})  # type: ignore[arg-type]

    assert not inventory.poolable
    assert [item.key for item in inventory.items] == [("chickpea", "id"), ("lentil", "id"), ("lentil", "parameter_ood")]
    assert [(item.label, reason) for item, reason in inventory.unavailable] == [("Material OOD", "Rolle ist nicht verfügbar")]
    choices = selection.resolve_choices(inventory, [("lentil", "parameter_ood"), ("lentil", "id")], pooled=False)

    contexts = datasets.transient_contexts(session, choices)  # type: ignore[arg-type]

    assert [(context.label, context.frame_names, context.case_ids) for context in contexts] == [
        ("lentil, ID", ("ID",), ("c1",)),
        ("lentil, Parameter OOD", ("OOD",), ("c3", "c4")),
    ]
    with pytest.raises(ValueError, match="unavailable"):
        datasets.transient_contexts(session, selection.resolve_choices(inventory, [("", "material_ood")], pooled=False))  # type: ignore[arg-type]
