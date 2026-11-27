"""Protect how the Evaluation views name runs, roles, cases and table columns.

Responsibilities:
  - Check that a run label prefers a compact run-name token over a long Dataset alias, and that the run-level views of a
    single steady model name the run by its label, never by the role of a context
  - Check that a transient frame name and a series of several runs name each run and role once, by display names, and that a
    dataset reads as its material and role in one unit («Chickpea, ID»)
  - Check that table keys read as headings with display units and basis, that a shown table names role, material, family,
    work-unit and status tokens, leaves a missing cell blank and keeps an identity exact
  - Check that the configuration shows the training Dataset short with its exact identity among the details
  - Check that a generation case id reads as its case number
  - Check that the summary and work tables write a physics-informed run as PI-U-NO, name the units of the physical group
    errors, show work counts as whole numbers and leave out a role the row label already names
  - Check that the configuration table shows one spectral-modes row, tokens as names, and absent values as a dash or no row

Design principles:
  - Inputs are small synthetic runs, frames and rows; the views' builders run unchanged

This module does NOT:
  - Check plot geometry, colours or the numbers of any metric
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ipywidgets as widgets
import pandas as pd
import pytest
from support.steady_evaluation import steady_frame

from src import analysis
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.evaluation import evaluation_workspace as workspace
from src.analysis.presentation import analysis_display_labels as display_labels

_DEMO = "lentil-train32-val8-test8+chickpea-train32-val8-test8+kidney-bean-ood16__ref-range__sobol-seed-9200"
_SMOKE = "lentil-train2-val1-test1+chickpea-train2-val1-test1+kidney-bean-train2-val1-test1__ref-range__sobol-seed-9250"


def _discovered_a0(run_name: str, dataset_id: str) -> Any:
    """Return the fields of a discovered FNO A0 run of seed 9 that its short label reads (a stand-in for the discovery record)."""
    return SimpleNamespace(
        model_family="fno",
        model_label="fno_m64x80",
        stage="A0",
        seed=9,
        pair_label=None,
        config_label="architecture_comparison/fno_m64x80",
        dataset_id=dataset_id,
        run_name=run_name,
        run_dir=Path("/runs") / dataset_id / "a0",
    )


def test_two_a0_runs_on_different_datasets_are_told_apart_by_their_compact_run_name_token() -> None:
    """The run names differ in one token where the Dataset aliases differ at length: the label takes that token, not the alias."""
    runs = (
        _discovered_a0("fno_m64x80__seed-9__ep-100__k-16__a0", _DEMO),
        _discovered_a0("fno_m64x80__seed-9__ep-2__k-16__a0", _SMOKE),
    )

    assert workspace._run_labels(runs) == ("FNO · A0 · seed 9 · ep-100", "FNO · A0 · seed 9 · ep-2")


def _role_frame(directory: Path, case_ids: tuple[int, int], *, split_role: str) -> pd.DataFrame:
    """Return one two-case steady frame of one run's split role with the given case identities."""
    directory.mkdir()
    frame, _grid = steady_frame(directory)
    frame["case_index"] = list(case_ids)
    frame.attrs["artifact_provenance"]["split_role"] = split_role
    frame.attrs["dataset_role"] = split_role
    return frame


def test_the_run_level_views_of_a_single_steady_model_name_the_run_by_its_label(tmp_path: Path) -> None:
    """One run with an ID and a material-OOD context: the break-even names the run by its label, never by a context's role."""
    label = "FNO · seed 9"
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets={label: _role_frame(tmp_path / "id", (1, 2), split_role="eval")}),
        panel.EvaluationContext(
            key="material_ood", label="Material OOD", datasets={label: _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")}
        ),
    ]
    registry = panel._build_sections(panel._SteadyScope(contexts, comparison=False, refused_roles={}), comparison=False)

    shown = {title: factory for title, factory, _stem in registry["runtime"][0]}["Break-even"]()

    assert isinstance(shown, widgets.HTML)  # the fixture records no training work, so the message names the run and what it lacks
    assert f"{label} (no " in shown.value
    assert "ID (no " not in shown.value


def test_transient_frames_and_their_series_name_each_run_and_role_once_by_display_name() -> None:
    """A frame name is the run label and the role's display name; a series of several runs names its run, then material and role once."""
    runs = {}
    rows = []
    for run in ("FNO · B · seed 9", "FNO · A1 · seed 9"):
        for role, material in (("id", "lentil"), ("material_ood", "kidney_bean")):
            frame = analysis.evaluation.workflow._frame_name(run, role)
            runs[frame] = run
            rows.append({"frame": frame, "material_family": material, "dataset_role": role, "model_kind": "fno", "checkpoint_sha256": run})

    labels = list(analysis.evaluation.plots.transient._summary_series(pd.DataFrame(rows), runs))

    assert labels == [
        "FNO · B · seed 9 · Lentil, ID",
        "FNO · B · seed 9 · Kidney bean, Material OOD",
        "FNO · A1 · seed 9 · Lentil, ID",
        "FNO · A1 · seed 9 · Kidney bean, Material OOD",
    ]


def test_a_dataset_reads_its_material_and_role_as_one_unit() -> None:
    """«Kidney bean, Material OOD»: the material's display name, a comma and the role's display name, which every view shares."""
    assert display_labels.material_role_display_label("kidney_bean", "material_ood") == "Kidney bean, Material OOD"
    assert analysis.presentation.dataset_selection.dataset_label("chickpea", "id") == "Chickpea, ID"


@pytest.mark.parametrize(
    ("key", "heading"),
    [
        ("normalized_rmse_w_surf", "Normalized RMSE w_surf (water density) [-]"),  # dimensionless, on the stored water-density basis
        ("physical_rmse_w_gr [kg/m^3]", "Physical RMSE w_gr (water density) [kg/m³]"),
        ("physical_relative_l2_T", "Physical relative L2 T [-]"),
        ("physical_relative_l2_unavailable_cases", "Physical relative L2 unavailable cases"),  # a count has no unit
        ("solver_speedup Σ comsol_solver_s", "Solver speedup Σ COMSOL solver [s]"),
        ("optimisation_seconds", "Optimization time [s]"),  # the spelling of the configuration headings
        ("normalized_drying_group_macro_rmse", "Drying group macro RMSE [-]"),
    ],
)
def test_table_keys_read_as_headings_with_display_units(key: str, heading: str) -> None:
    """Words in sentence case with acronyms, a field key as one symbol with its basis, a unit in display form, a declared heading as declared."""
    assert display_labels.table_heading(key) == heading


def test_a_summary_row_with_an_availability_reason_keeps_its_metric_and_leaves_missing_cells_blank() -> None:
    """The row names why it lacks values once; a metric it has stays shown, a missing one is blank, never ``None`` or ``nan``."""
    row = {
        "frame": "FNO · ID",
        "material_family": "lentil",
        "mode": "autonomous_full",
        "requested_horizon": "full",
        "scope": "cumulative",
        "contributing_record_count": 3,
        "contributing_case_count": 3,
        "unavailable_case_count": 0,
        "finite_value_count": 5,
        "required_value_count": 8,
        "normalized_drying_group_macro_rmse": None,
        **{f"normalized_rmse_{field}": math.nan for field in analysis.evaluation.transient_artifact.STATE_ORDER},
        "physical_relative_l2": math.nan,
        "physical_relative_h1": None,
    }
    row["normalized_rmse_T"] = 0.25
    session = SimpleNamespace(dataset_dataframe=lambda: pd.DataFrame([row]))

    shown = analysis.evaluation.plots.run_summary.display_table(panel._transient_summary_table(session)).iloc[0]  # type: ignore[arg-type]

    assert shown["Availability"] == "unavailable (5 finite of 8 required values)"
    assert shown["Normalized RMSE T [-]"] == 0.25
    assert shown["Normalized RMSE w_surf (water density) [-]"] == ""
    assert shown["Drying group macro RMSE [-]"] == ""
    assert shown["Physical relative H1 [-]"] == ""


def test_a_shown_work_table_names_its_tokens_and_keeps_the_run_name_exact() -> None:
    """Budget unit, unit kind, status and time coverage tokens read as words; the run name, an identity, stays exactly as stored."""
    work = pd.DataFrame(
        {
            "budget_unit": ["supervised_transitions"],
            "unit_kind": ["transition"],
            "status": ["no_eligible_checkpoint"],
            "time_coverage": ["unknown_attempts"],
            "run_name": ["fno__seed-9"],
        }
    )

    shown = analysis.evaluation.plots.run_summary.display_table(work)

    assert shown.to_dict("records") == [
        {
            "Budget unit": "Supervised transitions",
            "Unit kind": "Transition",
            "Status": "No eligible checkpoint",
            "Time coverage": "Unknown attempts",
            "Run name": "fno__seed-9",
        }
    ]


def test_the_configuration_shows_the_training_dataset_short_and_its_exact_identity_in_the_details(tmp_path: Path) -> None:
    """The configured training Dataset reads short in the data group; its exact identity stands whole among the evidence details."""
    frame, _grid = steady_frame(tmp_path)
    frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_CONFIG_ATTR] = {
        "model": {"kind": "fno", "params": {}},
        "data": {"train_dataset": _DEMO},
        "loss": {"physics": {"enabled": False}},
    }

    row = analysis.evaluation.plots.run_summary.build_architecture_table({"FNO": frame}).loc["FNO"].to_dict()

    assert row["Data · Configured training Dataset"] == display_labels.compact_identity_label(_DEMO)
    assert len(row["Data · Configured training Dataset"]) < len(_DEMO)
    assert row["Runtime/evidence · Configured training Dataset ID"] == _DEMO


def test_the_optimizer_betas_read_as_an_ordered_pair_not_a_product(tmp_path: Path) -> None:
    """AdamW's two decay rates read «(0.9, 0.999)»; «x» stays reserved for spatial sizes such as the spectral modes."""
    frame, _grid = steady_frame(tmp_path)
    frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_CONFIG_ATTR] = {
        "model": {"kind": "fno", "params": {}},
        "optimizer": {"kind": "adamw", "betas": [0.9, 0.999]},
        "loss": {"physics": {"enabled": False}},
    }

    row = analysis.evaluation.plots.run_summary.build_architecture_table({"FNO": frame}).loc["FNO"].to_dict()

    assert row["Optimization · Betas"] == "(0.9, 0.999)"


def test_a_shown_table_names_role_material_and_family_tokens_and_keeps_its_numbers() -> None:
    """The builders keep raw tokens; the shown table writes their display names and a flag as Yes or No, the count unchanged."""
    table = pd.DataFrame(
        {"dataset_role": ["material_ood"], "material_family": ["kidney_bean"], "model_kind": ["uno"], "physics_enabled": [True], "sample_count": [16]}
    )

    shown = analysis.evaluation.plots.run_summary.display_table(table)

    assert shown.to_dict("records") == [
        {"Role": "Material OOD", "Material": "Kidney bean", "Model family": "U-NO", "Physics loss": "Yes", "Cases": 16}
    ]


def test_a_physics_informed_run_reads_pi_family_in_the_steady_summary_and_work_tables(tmp_path: Path) -> None:
    """The family of a physics-informed U-NO run reads «PI-U-NO» as in the configuration view; a data-loss run keeps «U-NO»."""
    plots = analysis.evaluation.plots.run_summary
    families = {}
    for physics_enabled in (True, False):
        frame = _role_frame(tmp_path / str(physics_enabled), (1, 2), split_role="eval")
        frame.attrs["artifact_provenance"]["model"].update(kind="uno", physics_enabled=physics_enabled)
        frame.attrs[analysis.evaluation.dataframe.COMPLETED_RUN_SUMMARY_ATTR]["model_kind"] = "uno"
        families[physics_enabled] = (
            plots.display_table(plots.build_run_summary_table({"Run": frame}))["Model family"].tolist(),
            plots.display_table(plots.build_frame_work_table({"Run": frame}, lambda _label, _frame: 0.1))["Model family"].tolist(),
        )

    assert families == {True: (["PI-U-NO"], ["PI-U-NO"]), False: (["U-NO"], ["U-NO"])}


def test_the_physical_group_errors_of_the_summary_name_their_units(tmp_path: Path) -> None:
    """Pressure reads in Pa and velocity in m/s, as the physical field errors do; the normalized group errors stay dimensionless."""
    plots = analysis.evaluation.plots.run_summary
    frame, _grid = steady_frame(tmp_path)

    headings = set(plots.display_table(plots.build_run_summary_table({"Run": frame})).columns)

    assert {"Physical group RMSE pressure [Pa]", "Physical group RMSE velocity [m/s]"} <= headings
    assert {"Normalized group RMSE pressure [-]", "Normalized group RMSE velocity [-]"} <= headings


def test_the_work_table_shows_counts_as_whole_numbers_and_seconds_as_measured() -> None:
    """A count of 249 700 transitions reads «249,700», never «2.497e+05», a large count exactly; seconds and fractions stay as measured."""
    work = pd.DataFrame(
        {
            "optimizer_steps": [249700.0],
            "supervised_transitions": [249700],
            "full_set_transitions": [2**53 + 1],
            "overrun": [0.25],
            "total_seconds": [1234.5678],
            "attempted_seconds": [4321.0],
        },
        index=pd.Index(["Run"]),
    )

    html = analysis.evaluation.plots.run_summary.show_work_table(work).children[1].value  # type: ignore[attr-defined]

    assert html.count("249,700") == 2
    assert "9,007,199,254,740,993" in html
    assert "e+05" not in html
    assert "1235" in html
    assert "0.25" in html
    assert "4,321" not in html


def test_the_runtime_table_leaves_out_the_role_only_when_every_row_label_already_names_it(tmp_path: Path) -> None:
    """Rows labelled «ID» need no Role column; a run label that does not name the role keeps it, and a mixed table blanks the repeat."""
    plots = analysis.evaluation.plots.run_summary
    frame, _grid = steady_frame(tmp_path)

    named = plots.plot_runtime_table(datasets={"ID": frame}).children[1].value  # type: ignore[attr-defined]
    unnamed = plots.plot_runtime_table(datasets={"FNO · seed 9": frame}).children[1].value  # type: ignore[attr-defined]
    mixed = plots.plot_runtime_table(datasets={"ID": frame, "FNO · seed 9": frame}).children[1].value  # type: ignore[attr-defined]

    assert ">Role<" not in named
    assert ">Role<" in unnamed
    assert ">Role<" in mixed
    assert mixed.count(">ID<") == 2  # the «ID» row label and the role of the run-labelled row, not a third repeat


def test_every_run_shows_one_spectral_modes_row_whatever_the_architecture_stores() -> None:
    """A grid pair stored as one list or as x and y modes reads «Spectral modes» in one form; an unnamed key reads in sentence case."""
    plots = analysis.evaluation.plots.run_summary

    stored_as_list = dict(plots.architecture_items({"n_modes": [64, 80], "rno_skip": "soft-gating"}))
    stored_as_pair = dict(plots.architecture_items({"modes_x": 64, "modes_y": 80, "mode_ratio": 0.5}))

    assert stored_as_list["Architecture · Spectral modes"] == stored_as_pair["Architecture · Spectral modes"] == "64 x 80"
    assert not {key for key in stored_as_pair if "modes x" in key or "modes y" in key}
    assert "Architecture · RNO skip" in stored_as_list
    assert "Architecture · U-NO mode ratio" in stored_as_pair


def test_the_configuration_shows_tokens_as_names_and_absent_values_as_a_dash_or_no_row() -> None:
    """A row every run lacks is omitted; a value only some runs lack reads as a dash; a stored token reads as its name, a flag as Yes or No."""
    raw = pd.DataFrame(
        {
            "Architecture · Positional embedding": [None, None],
            "Architecture · Channel-MLP skip": [None, "linear"],
            "Optimization · Optimizer": ["adamw", "adamw"],
            "Optimization · Data loss": ["mse", "mse"],
            "Runtime/evidence · Requested device": ["cuda", "cpu"],
            "Physics-informed loss · Enabled": [True, False],
        },
        index=pd.Index(["A", "B"], name="label"),
    )

    shown = analysis.evaluation.plots.run_summary.display_configuration_table(raw)

    assert {key: list(row) for key, row in shown.iterrows()} == {
        "Architecture · Channel-MLP skip": ["\N{EN DASH}", "Linear"],
        "Optimization · Optimizer": ["AdamW", "AdamW"],
        "Optimization · Data loss": ["MSE", "MSE"],
        "Runtime/evidence · Requested device": ["CUDA", "CPU"],
        "Physics-informed loss · Enabled": ["Yes", "No"],
    }  # no row for the positional embedding, which no run has


def test_a_frame_names_only_its_three_stored_roles_and_any_other_or_missing_role_reads_unspecified() -> None:
    """The stored split roles read as the shared role labels; an unknown value, a missing one and ``pd.NA`` are never given a label."""
    frame = pd.DataFrame()
    shown = {}
    for role in ("eval", "material_ood", "parameter_ood", "legacy_role", None, pd.NA):
        frame.attrs = {} if role is None else {"dataset_role": role}
        shown[str(role)] = analysis.evaluation.dataframe.dataset_role(frame)
    assert shown == {
        "eval": "ID",
        "material_ood": "Material OOD",
        "parameter_ood": "Parameter OOD",
        "legacy_role": "unspecified",
        "None": "unspecified",
        "<NA>": "unspecified",
    }


def test_a_generation_case_id_reads_as_its_case_number_and_any_other_id_stays_whole() -> None:
    """``case_0048`` reads ``case 48``, the form of every case title; an id of another form is never cut."""
    assert display_labels.case_display_label("case_0048") == "case 48"
    assert display_labels.case_display_label("lentil__case_0048__ab12") == "lentil__case_0048__ab12"
