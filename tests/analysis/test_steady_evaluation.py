"""Protect steady Evaluation maps, metrics, timing, and shared widget state.

Responsibilities:
  - Check steady maps, metrics, timing bindings, exact grids, and shared controls
  - Check that compared frames record one valid analysis producer within and across roles
  - Check that titles name runs only by explicit short labels, none where a run was selected by its path alone
  - Check that a pooled role is one frame per material, drawn as the equal-weight mean with each material as a marker, in the two views that pool
  - Check that pooling refuses materials on different grids, names an undefined scoreboard ratio with its raw values, and never names the
    pooled capacity point the artifact objective
  - Check that a capacity drawn only by a material marker stays on the capacity axis and that point keys show the marker their
    models are drawn with, a neutral circle beside a material key where a model's points carry several, and keep a model named
    like a material apart from that material in the capacity and the Pareto key
  - Check that only the ranked error columns of the steady tables are shaded

Design principles:
  - Matched-case grid identity is checked independently of lazy view admission
  - The producer check reads recorded provenance only, never the running analysis code

This module does NOT:
  - Train steady models or provide transient evaluation behavior
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
from support.steady_evaluation import steady_frame

from src import analysis, common, domain
from src.analysis.evaluation import evaluation_artifact_loader as artifact_loader
from src.analysis.evaluation import evaluation_case as cases
from src.analysis.evaluation import evaluation_dataframe as dataframe
from src.analysis.evaluation import evaluation_datasets as dataset_cuts
from src.analysis.evaluation import evaluation_panel as panel
from src.analysis.evaluation import evaluation_prediction_validity as validity_contract
from src.analysis.presentation import analysis_error_diagnostics as error_diagnostics
from src.analysis.presentation import analysis_visual_semantics as visual_semantics
from src.analysis.presentation import dataset_selection as selection

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.patches import Rectangle

_CASE_COUNT = 2
_MANY_MODELS = tuple(f"model {model}" for model in range(80))  # more lines than a 3 in panel holds even at the smallest font
_ORIGIN_MAIN_STEADY_BASELINE_COMMIT = "bf8058aefaa960dc34ffac569a6f67b59de1bf5c"


def _numeric_array(values: Any) -> np.ndarray:
    """Narrow third-party plotting array protocols for numeric assertions."""
    return np.asarray(values, dtype=float)


def _ecdf_steps(line: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the sorted values and CDF levels of an ECDF line, after checking its leading rise at the smallest value."""
    values, levels = _numeric_array(line.get_xdata()), _numeric_array(line.get_ydata())
    assert values[0] == values[1]
    assert levels[0] < levels[1]
    return values[1:], levels[1:]


def _medians(axis: Any) -> list[Any]:
    """Return the lines of a spectral panel without the edges of its q10-q90 bands, the thin lines with open markers."""
    return [line for line in axis.lines if line.get_markerfacecolor() != "none"]


def _contexts_of(datasets: Mapping[str, pd.DataFrame], *, comparison: bool) -> list[Any]:
    """Return the contexts of ID-role frames: one labelled context per frame (single model) or one context of models (comparison)."""
    if comparison:
        return [panel.EvaluationContext(key="id", label="ID", datasets=dict(datasets))]
    return [panel.EvaluationContext(key="id", label=label, datasets={label: frame}) for label, frame in datasets.items()]


def _scope(datasets: Mapping[str, pd.DataFrame], *, comparison: bool, refused: Mapping[str, str] | None = None) -> Any:
    """Return the dataset scope of a panel over ID-role frames."""
    return panel._SteadyScope(_contexts_of(datasets, comparison=comparison), comparison=comparison, refused_roles=refused or {})


def _displayed(frames: Mapping[str, pd.DataFrame], *, dataset: str | None = None) -> dataset_cuts.DisplayFrames:
    """Label ``frames`` as the models of one dataset (``dataset`` given) or else as the datasets of one model."""
    return dataset_cuts.DisplayFrames(frames, {key: (dataset, key) if dataset else (key, "Model") for key in frames})


def _figure_texts(figure: Any) -> list[str]:
    """Return the text of the figure title and note, every panel text and every legend entry and title, whitespace collapsed (a long note wraps)."""
    texts = [artist.get_text() for artist in figure.artists if hasattr(artist, "get_text")]
    texts += [text.get_text() for axis in figure.axes for text in axis.texts]
    for legend in (*figure.legends, *(axis.get_legend() for axis in figure.axes)):
        if legend is not None:
            texts += [text.get_text() for text in (legend.get_title(), *legend.get_texts())]
    return [" ".join(text.split()) for text in texts]


def _widget_tree(widget: object) -> tuple[Any, ...]:
    """Return one widget and every nested child in deterministic preorder."""
    return (widget, *(item for child in getattr(widget, "children", ()) for item in _widget_tree(child)))


def _record_shows(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every item a plot slot shows, in order, while the slot still builds its elements."""
    shown: list[object] = []
    show = analysis.ui.panel.PlotSlot.show

    def record(slot: Any, items: Sequence[object]) -> None:
        shown.extend(items)
        show(slot, items)

    monkeypatch.setattr(analysis.ui.panel.PlotSlot, "show", record)
    return shown


def _taken(shown: list[object]) -> list[object]:
    """Return the items shown since the last call and forget them."""
    items = list(shown)
    shown.clear()
    return items


def _record_payload(frame: pd.DataFrame, path: Path) -> None:
    """Bind one payload file to its current digest in the frame's recorded output manifest."""
    entries = frame.attrs["artifact_provenance"]["outputs"]["npz"]
    entries[:] = [entry for entry in entries if entry["path"] != path.name]
    entries.append({"path": path.name, "sha256": common.serialization.cached_file_sha256(path)})


def test_steady_table_and_plots_show_an_unavailable_aggregate_by_its_support_counts(tmp_path: Path) -> None:
    """Cells, scoreboard ticks and the Pareto notice carry the counts, both titles still name Model B; per-case statistics drop a non-finite case."""
    frame, _grid = steady_frame(tmp_path)
    peer = _steady_peer(frame)
    aggregate = peer.attrs[dataframe.PRIMARY_OBJECTIVE_ID]
    aggregate.update(available=False, value=None, field_metrics=None, relative_unavailable_cases=None, group_statistics=None, finite_value_count=0)
    aggregate["nonfinite_value_count"] = aggregate["required_value_count"]
    text = validity_contract.unavailable_text(aggregate)
    datasets = {"Model A": frame, "Model B": peer}
    plots = analysis.evaluation.plots.run_summary

    table = plots.build_run_summary_table(datasets)
    assert (table.loc["Model B", "normalized_group_macro_rmse"], table.loc["Model B", "normalized_rmse_u"]) == (text, text)
    assert table.loc["Model A", "normalized_group_macro_rmse"] == pytest.approx(frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"])
    assert text in plots.plot_run_summary_table(datasets=datasets).children[1].value

    models = _displayed(datasets, dataset="ID")
    scoreboard = plots.plot_relative_comparison_scoreboard(datasets=models)
    # Only model A has a bar per diagnostic; model B is named with its dataset and support counts in the note instead.
    assert len(scoreboard.axes[0].patches) == len(scoreboard.axes[0].get_xticklabels())
    assert any(f"Model B on ID: {text}" in entry for entry in _figure_texts(scoreboard))
    assert _figure_texts(scoreboard)[0].endswith(" (Model A vs Model B)")  # a compared model without a bar stays named
    plt.close(scoreboard)

    pareto = plots.plot_accuracy_physics_pareto(datasets=models)
    assert [len(axis.collections) for axis in pareto.axes[:3]] == [1, 1, 1]
    assert any(text in note for note in _figure_texts(pareto))
    assert _figure_texts(pareto)[0].endswith(" (Model A vs Model B)")
    for axis in pareto.axes[:3]:  # dimensionless units read [-], never [1]
        assert axis.get_xlabel() == "Normalized task error [-]"
        assert axis.get_ylabel().endswith("[-]")
    plt.close(pareto)

    peer.loc[peer.index[0], "flux_balance_mse"] = np.nan
    physics = analysis.evaluation.plots.physical_consistency.build_physical_consistency_summary_table(datasets)
    assert physics.loc["Model B", "flux_balance_mse median"] == pytest.approx(frame["flux_balance_mse"].iloc[1])

    peer.loc[peer.index[0], "normalized_rmse_p"] = np.nan
    sensitivity = analysis.evaluation.plots.sensitivity_capacity
    heatmap = sensitivity.plot_metadata_error_heatmap(datasets=models, max_cases=_CASE_COUNT)
    np.testing.assert_allclose(_numeric_array(heatmap.axes[0].images[0].get_array()), 1.0)
    assert np.isnan(_numeric_array(heatmap.axes[1].images[0].get_array())[:, 0]).all()
    plt.close(heatmap)
    plt.close(sensitivity.plot_metadata_error_trends(datasets=models, max_cases=_CASE_COUNT, channels=("p",)))


def _filled_cells(table: widgets.VBox) -> dict[str, int]:
    """Return each column heading of a rendered table with its number of filled cells, which Styler lists as CSS selectors."""
    rules, body = cast("widgets.HTML", table.children[1]).value.split("</style>")
    headings = re.findall(r'class="col_heading level0 col(\d+)" >([^<]*)</th>', body)
    return {heading: len(re.findall(rf"_row\d+_col{index}\b", rules)) for index, heading in headings}


def _table_html(root: widgets.Widget) -> str:
    """Return the HTML of every table in ``root``, also of tables grouped in rows."""
    own = root.value if isinstance(root, widgets.HTML) else ""
    return " ".join((own, *(_table_html(child) for child in getattr(root, "children", ()))))


def test_only_the_ranked_error_columns_of_the_steady_tables_are_shaded(tmp_path: Path) -> None:
    """
    The global summary fills both models' cells of its error columns and none of its identity or count columns.

    The work, model-family configuration and extreme-parameter tables rank nothing, so none of their cells is filled, although
    the two models differ in their trainable parameters.
    """
    frame, _grid = steady_frame(tmp_path)
    datasets = {"Model A": frame, "Model B": _steady_peer(frame)}
    plots = analysis.evaluation.plots
    heading = analysis.presentation.display_labels.table_heading
    errors = ("normalized_group_macro_rmse", "normalized_rmse_u", "physical_relative_l2")
    descriptive = ("trainable_parameters", "sample_count", "physical_relative_l2_unavailable_cases")

    filled = _filled_cells(plots.run_summary.plot_run_summary_table(datasets=datasets))
    assert {column: filled[heading(column)] for column in (*errors, *descriptive)} == {**dict.fromkeys(errors, 2), **dict.fromkeys(descriptive, 0)}
    for table in (
        plots.run_summary.plot_work_table(datasets=datasets),
        plots.run_summary.plot_architecture_table(datasets=datasets),
        plots.samples_outliers.plot_input_extremes_table(datasets=datasets),
    ):
        assert "background-color" not in _table_html(table)


def test_steady_comparison_checks_every_matched_case_grid_after_lazy_panel_admission(tmp_path: Path) -> None:
    """A later case's changed coordinates cannot enter scoreboards or Pareto."""
    frame, _grid = steady_frame(tmp_path)
    alternate = frame.copy(deep=True)
    original_path = Path(str(alternate.iloc[-1]["npz_path"]))
    with np.load(original_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["coordinates"] = arrays["coordinates"].copy()
    arrays["coordinates"][0] += 0.01
    alternate_path = tmp_path / "alternate_case_2.npz"
    np.savez_compressed(alternate_path, **arrays)
    _record_payload(alternate, alternate_path)
    alternate.loc[alternate.index[-1], "npz_path"] = str(alternate_path)
    datasets = {"A": frame, "B": alternate}
    dataframe.validate_comparison(datasets, verify_grid=False)
    with pytest.raises(dataframe.ComparisonCompatibilityError, match=r"case.*coordinate_digest"):
        dataframe.validate_comparison(datasets)


def test_comparison_cuts_of_models_with_different_split_positions_pass_the_grid_admission(tmp_path: Path) -> None:
    """Cuts holding the same case are matched by case and source identity, whatever split-local position each model saved it at."""
    frame, _grid = steady_frame(tmp_path)  # model A saves [case-a, case-b]
    extra_path = tmp_path / "extra_case.npz"
    shutil.copyfile(Path(str(frame.iloc[0]["npz_path"])), extra_path)
    extra = frame.iloc[[0]].copy()
    extra["simulation_case_id"], extra["source_index"], extra["case_index"], extra["npz_path"] = "case-c", 2, 3, str(extra_path)
    peer = _steady_peer(frame)  # model B saves [case-b, case-c]
    peer_table = pd.concat([frame.iloc[[1]], extra], ignore_index=True)
    peer_table["split_local_index"] = [0, 1]
    peer_table.attrs = peer.attrs
    _record_payload(peer_table, extra_path)
    models = {"A": frame, "B": peer_table}
    inventory = dataset_cuts.comparison_inventory({"id": models}, refused={})
    choice = selection.resolve_choices(inventory, [("lentil", "id")], pooled=False)[0]

    cuts = dataset_cuts.comparison_frames(models, choice)

    assert [cut["simulation_case_id"].tolist() for cut in cuts.values()] == [["case-b"], ["case-b"]]
    assert [cut["split_local_index"].tolist() for cut in cuts.values()] == [[1], [0]]
    dataframe.validate_comparison(cuts)


def test_steady_comparison_refuses_runs_with_different_role_membership(tmp_path: Path) -> None:
    """Steady comparisons refuse disagreeing membership instead of intersecting it, as transient ones do."""
    frame, _grid = steady_frame(tmp_path)
    narrowed = frame.iloc[:-1].copy()
    narrowed.attrs = copy.deepcopy(frame.attrs)
    narrowed.attrs["artifact_provenance"]["dataset"]["case_ids_by_source"] = [["case-a"]]

    def loaded(table: pd.DataFrame) -> Any:
        artifacts = {"id": SimpleNamespace(frame=table)}
        return SimpleNamespace(artifact=artifacts.get, run_dir=tmp_path)

    workflow = analysis.evaluation.workflow
    workflow._validate_shared_membership((loaded(frame), loaded(frame.copy())), roles=("id",))
    with pytest.raises(ValueError, match="dataset membership"):
        workflow._validate_shared_membership((loaded(frame), loaded(narrowed)), roles=("id",))


@pytest.mark.parametrize("peer_role", ["eval", "material_ood"])
@pytest.mark.parametrize(
    ("producer", "message"),
    [
        ({"analysis_sha256": "0" * 64}, "different analysis implementations"),
        (None, "cannot be compared: Artifact producer must be exactly"),
        ({"analysis_sha256": "D" * 64}, "cannot be compared: Artifact producer must be exactly"),
    ],
    ids=["different", "missing", "malformed"],
)
def test_steady_comparison_requires_one_recorded_analysis_producer(
    tmp_path: Path,
    peer_role: str,
    producer: dict[str, str] | None,
    message: str,
) -> None:
    """Frames of one comparison, within one role or across ID and OOD, must record the same valid analysis producer."""
    frame, _grid = steady_frame(tmp_path)
    peer = _steady_peer(frame)
    peer.attrs["artifact_provenance"]["split_role"] = peer_role
    dataframe.validate_comparison({"A": frame, "B": peer}, verify_grid=False)

    if producer is None:
        del peer.attrs["artifact_provenance"]["producer"]
    else:
        peer.attrs["artifact_provenance"]["producer"] = producer
    with pytest.raises(dataframe.ComparisonCompatibilityError, match=message):
        dataframe.validate_comparison({"A": frame, "B": peer}, verify_grid=False)


def test_provenance_with_a_non_mapping_physics_section_is_refused(tmp_path: Path) -> None:
    """A present physics section must be a mapping, so a malformed one never passes as complete provenance."""
    frame, _grid = steady_frame(tmp_path)
    provenance = copy.deepcopy(frame.attrs["artifact_provenance"])
    arguments = {
        "task_id": frame.attrs["task_id"],
        "output_fields": tuple(frame.attrs["output_fields"]),
        "output_units": tuple(frame.attrs["output_units"]),
        "aggregate": frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID],
    }
    dataframe._validated_provenance(provenance, **arguments)

    provenance["physics"] = "invalid"
    with pytest.raises(TypeError, match=r"provenance\.physics must be a mapping"):
        dataframe._validated_provenance(provenance, **arguments)


def test_exact_grid_cache_rejects_changed_payload_witness(tmp_path: Path) -> None:
    """A saved grid digest never excuses a changed source payload."""
    frame, _grid = steady_frame(tmp_path)
    alternate = frame.copy(deep=True)
    datasets = {"A": frame, "B": alternate}
    dataframe.validate_comparison(datasets)
    path = Path(str(frame.iloc[-1]["npz_path"]))
    witness = path.stat()
    os.utime(path, ns=(witness.st_atime_ns, witness.st_mtime_ns + 1_000_000_000))
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="changed after exact grid admission"):
        dataframe.validate_comparison(datasets)


def test_steady_payload_changed_after_load_is_refused_by_digest_at_case_open_and_grid_admission(tmp_path: Path) -> None:
    """Payload bytes are verified against the recorded digest before any byte is parsed."""
    frame, _grid = steady_frame(tmp_path)
    path = Path(str(frame.iloc[0]["npz_path"]))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Steady case payload digest changed"):
        cases.load_case(frame, 0)
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="Steady case payload digest changed"):
        dataframe.validate_comparison({"A": frame, "B": _steady_peer(frame)})


def test_mixed_case_grids_fail_single_and_identically_mixed_model_comparison(tmp_path: Path) -> None:
    """A scoreboard cannot accept a changing grid within one model's cases."""
    frame, _grid = steady_frame(tmp_path)
    mixed = frame.copy(deep=True)
    source = Path(str(mixed.iloc[-1]["npz_path"]))
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["coordinates"] = arrays["coordinates"].copy()
    arrays["coordinates"][0] += 0.02
    changed = tmp_path / "mixed_case_2.npz"
    np.savez_compressed(changed, **arrays)
    _record_payload(mixed, changed)
    mixed.loc[mixed.index[-1], "npz_path"] = str(changed)
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="mixes incompatible grids"):
        dataframe.validate_comparison({"A": mixed})
    with pytest.raises(dataframe.ComparisonCompatibilityError, match="mixes incompatible grids"):
        dataframe.validate_comparison({"A": mixed, "B": _steady_peer(mixed)})


def _steady_peer(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one exact-case peer with a distinct persisted model identity."""
    peer = frame.copy(deep=True)
    peer.attrs = copy.deepcopy(frame.attrs)
    provenance = peer.attrs["artifact_provenance"]
    provenance["run"].update(
        {
            "name": "steady-peer",
            "effective_config_digest": "f" * 64,
            "best_checkpoint_sha256": "1" * 64,
        }
    )
    provenance["model"].update(
        {
            "parameter_counts": {"total": 16, "trainable": 16},
            "architecture": {"modes": 8},
        }
    )
    return peer


def test_steady_maps_metrics_timing_and_task_gating(tmp_path: Path) -> None:
    """Use exact fixture arrays for steady case, aggregate, timing, and gating checks."""
    frame, x_grid = steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    case = cases.load_case(frame, 0)
    fields = analysis.evaluation.presentation.display_fields(frame)
    pressure = next(field for field in fields if field.key == "p")
    np.testing.assert_allclose(
        analysis.evaluation.presentation.case_field(case, pressure, source="reference"),
        0.5 * x_grid,
    )
    np.testing.assert_allclose(
        analysis.evaluation.presentation.case_field(case, pressure, source="prediction"),
        x_grid,
    )
    assert pressure.unit == "Pa"

    with analysis.evaluation.session.EvaluationSession(datasets) as session:
        summary = session.full_summary(frame).require_spatial()
        pressure_index = summary.grid.fields.index("p")
        absolute_error_mean = summary.absolute_error_mean
        assert absolute_error_mean is not None
        np.testing.assert_allclose(absolute_error_mean[pressure_index], 0.75 * x_grid)
        aggregate_figure = analysis.evaluation.plots.error_behavior.plot_mean_error_maps(
            datasets=datasets,
            max_cases=2,
            error_mode="MAE",
        )
    assert "Mean absolute error maps" in aggregate_figure.get_suptitle()
    plt.close(aggregate_figure)

    expected_pressure_rmse = float(np.sqrt(0.625 * np.mean(x_grid**2)))
    aggregate = frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]
    assert aggregate["field_metrics"]["normalized_rmse_p"] == pytest.approx(expected_pressure_rmse)

    runtime = analysis.evaluation.plots.run_summary.build_runtime_table(datasets).iloc[0]
    assert (runtime["solver_speedup [-]"], runtime["solver_speedup paired cases"], runtime["solver_speedup unavailable cases"]) == (
        pytest.approx(4.0 / 0.1),
        1,
        1,
    )
    assert runtime["process_speedup [-]"] == pytest.approx(50.0 / 0.3)
    registry = panel._build_sections(_scope(datasets, comparison=False), comparison=False)
    assert "runtime" in registry
    assert not {"spatial", "rollout", "process", "outcomes"}.intersection(registry)
    titles = tuple(title.lower() for entries, _section_title in registry.values() for title, _factory, _stem in entries)
    assert not any(token in item for item in titles for token in ("trajectory", "rollout horizon", "physical time"))


def _levels(axis: Any) -> np.ndarray:
    """Return the discrete colour levels of the map drawn first on ``axis``."""
    return np.asarray(axis.collections[0].norm.boundaries)


def _round_steps(levels: np.ndarray) -> bool:
    """Return whether ``levels`` are 9 to 13 equal steps of 1, 2, 2.5 or 5 times a power of ten."""
    steps = np.diff(levels)
    mantissa = steps[0] / 10.0 ** np.floor(np.log10(steps[0]))
    return bool(np.allclose(steps, steps[0]) and np.isclose(mantissa, (1.0, 2.0, 2.5, 5.0)).any() and 9 <= steps.size <= 13)


def test_the_steady_sample_overview_draws_every_learned_field_on_round_discrete_levels(tmp_path: Path) -> None:
    """
    Rows p, u, v by reference, prediction, error; every bar on round levels, each a tick, with its unit.

    The state levels cover the reference's 5th to 95th percentile and the prediction takes them, with an arrow where it
    passes them; the default error is signed on a symmetric blue-white-red scale; u and v are turbo while p keeps its
    colormap; one note stands under the title and only the error maps carry a mean |e| line; the title names the model by its display
    name, and only for a frame not keyed by its dataset label.
    """
    frame, _grid = steady_frame(tmp_path)
    plot = analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview
    case = cases.load_case(frame, 0)
    fields = [field for field in analysis.evaluation.presentation.display_fields(frame) if not field.is_magnitude]
    figure = plot(datasets={"lentil, ID": frame}, row_position=0)
    maps, slots = (np.asarray(axes, dtype=object).reshape(3, 3) for axes in (figure.axes[:9], figure.axes[9:]))
    assert [axis.get_title() for axis in maps[0, :2]] == ["Reference", "Prediction"]
    assert maps[0, 2].get_title() == "Signed error"
    row_labels = [text.get_text() for axis in maps[:, 0] for text in axis.texts if text.get_gid() == "channel-row-label"]
    assert row_labels == [rf"$\mathbf{{{name}}}$" for name in ("p", "u", "v")]
    assert [slot.get_ylabel() for slot in slots.flat] == [*("[Pa]",) * 3, *("[m/s]",) * 6]
    for column in range(3):
        for map_axis, slot in zip(maps[:, column], slots[:, column], strict=True):
            levels = _levels(map_axis)
            assert _round_steps(levels)
            # Every bar labels its round boundaries; the signed one labels its band centres 0, +-w, +-2w ... instead.
            np.testing.assert_allclose(slot.get_yticks(), (levels[:-1] + levels[1:]) / 2.0 if column == 2 else levels, atol=1e-9)
    for row, (reference_map, prediction_map, error_map) in enumerate(maps):
        np.testing.assert_array_equal(_levels(prediction_map), _levels(reference_map))
        low, high = np.percentile(analysis.evaluation.presentation.case_field(case, fields[row], source="reference"), (5.0, 95.0))
        levels = _levels(reference_map)
        assert levels[0] <= low < levels[1]
        assert high <= levels[-1]
        # The prediction passes the reference's top level exactly where its values do, and only there it gets an arrow.
        prediction = analysis.evaluation.presentation.case_field(case, fields[row], source="prediction")
        assert maps[row, 1].collections[0].colorbar.extend == ("max" if prediction.max() > levels[-1] else "neither")
        error_levels = _levels(error_map)
        assert error_levels[0] == pytest.approx(-error_levels[-1])
        assert error_map.collections[0].get_cmap().name == visual_semantics.field_visual_semantics("error", role="signed_error").colormap
    figure.canvas.draw()
    for slot in slots.flat:  # exactly the decimals the tick step needs (none for a whole-number step)
        step = float(np.min(np.diff(slot.get_yticks())))
        decimals = analysis.presentation.field_maps.step_decimals(step)
        assert {len(text.get_text().partition(".")[2]) for text in slot.get_yticklabels()} == {decimals}
    assert "0.0" in [text.get_text() for text in slots[0, 2].get_yticklabels()]  # the white band is centred on 0
    assert [maps[row, 0].collections[0].get_cmap().name for row in range(3)] == [
        visual_semantics.field_visual_semantics("p").colormap,
        "turbo",
        "turbo",
    ]
    # Only the three error maps carry text: one mean |e| line each, over the drawn finite error, in the unit of their bar.
    assert not [text for axis in maps[:, :2].flat for text in axis.texts if text.get_gid() != "channel-row-label"]
    significant = analysis.evaluation.plots.layout.significant
    for row in range(3):
        (mean_line,) = (text.get_text() for text in maps[row, 2].texts if text.get_gid() != "channel-row-label")
        drawn = np.ma.filled(maps[row, 2].collections[0].get_array().astype(float), np.nan)
        assert mean_line == f"mean |e| = {significant(float(np.nanmean(np.abs(drawn))))} {slots[row, 2].get_ylabel().strip('[]')}"
    (note,) = (text.get_text() for text in figure.texts if text.get_text() != figure.get_suptitle())
    assert (
        note
        == "Colour scales: 5th\N{EN DASH}95th percentile, covered by round bands; prediction uses the reference scale; arrows mark values outside"
    )
    # No title carries a metric, whether or not the case has a relative L2; the case is named by its display case number.
    assert figure.get_suptitle() == "Lentil, ID — Case 1"
    plt.close(figure)
    second = plot(datasets={"lentil, ID": frame}, row_position=1)
    assert second.get_suptitle() == "Lentil, ID — Case 2"
    plt.close(second)

    named = plot(datasets={"uno_m64x64_h32_l7__fd": frame}, row_position=0)  # a bare mapping's key, such as a run name, is no model label
    assert named.get_suptitle() == "Lentil, ID — Case 1"
    plt.close(named)

    normalized = plot(datasets={"lentil, ID": frame}, row_position=0, error_mode=error_diagnostics.NORMALIZED_ABSOLUTE_ERROR)
    assert normalized.axes[2].get_title().startswith("Absolute error / train std")
    plt.close(normalized)


def test_state_levels_cover_the_percentile_range_tightly_and_are_never_cut_short() -> None:
    """The percentile range 0.05 to 0.95 gets ten bands of 0.1 from 0, reaching 0.95 with the last band; a tighter step beats a half-empty scale."""
    boundaries = analysis.evaluation.plots.samples_outliers._state_boundaries(np.linspace(0.0, 1.0, 101))
    assert boundaries is not None
    np.testing.assert_allclose(boundaries, np.linspace(0.0, 1.0, 11))
    # Case 42: p5 to p95 of p filled 52 % of a 0 to 550 Pa scale; now (step, bands, start, top) as below.
    state_boundaries = analysis.presentation.field_maps.state_boundaries
    for (lower, upper), (step, bands, start) in {
        (16.25, 304.9): (25.0, 13, 0.0),
        (-0.04119, 0.04502): (0.01, 10, -0.05),
        (0.0745, 0.2338): (0.02, 9, 0.06),
    }.items():
        np.testing.assert_allclose(state_boundaries(lower, upper), start + step * np.arange(bands + 1), atol=1e-12)


def test_exact_predictions_give_the_signed_and_normalized_mean_maps_a_round_band_scale_and_an_empty_pool_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-zero mean errors draw the scale of reach 1; the normalized mean map filters non-finite values and states an empty pool."""
    frame, _grid = steady_frame(tmp_path)
    frame.attrs["artifact_provenance"]["normalizer"].update(normalization=domain.tasks.spec.TRAIN_SCALING_KIND, fit_split="train", sha256="f" * 64)
    behavior = analysis.evaluation.plots.error_behavior
    full_array = behavior._full_array
    monkeypatch.setattr(behavior, "_full_array", lambda *args: np.zeros_like(full_array(*args)))
    for mode in (error_diagnostics.SIGNED_ERROR, error_diagnostics.NORMALIZED_ABSOLUTE_ERROR):
        figure = behavior.plot_mean_error_maps(datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT, error_mode=mode)
        # Reach 1: signed w = 0.2 (11 bands, span 2.2), normalized step 0.1 (10 bands, span 1).
        assert _levels(figure.axes[0]).size == (12 if mode == error_diagnostics.SIGNED_ERROR else 11)
        plt.close(figure)
    monkeypatch.setattr(behavior, "_full_array", lambda *args: np.full_like(full_array(*args), np.nan))
    figure = behavior.plot_mean_error_maps(
        datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT, error_mode=error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    )
    assert any("Unavailable" in text.get_text() for axis in figure.axes for text in axis.texts)
    plt.close(figure)

    def finite_with_one_nan(*args: Any) -> np.ndarray:
        values = np.ones_like(full_array(*args))
        values.flat[0] = np.nan  # finite and non-finite cells in one pool used to reach the map builder unfiltered
        return values

    monkeypatch.setattr(behavior, "_full_array", finite_with_one_nan)
    figure = behavior.plot_mean_error_maps(
        datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT, error_mode=error_diagnostics.NORMALIZED_ABSOLUTE_ERROR
    )
    assert 10 <= _levels(figure.axes[0]).size <= 14  # 9 to 13 round bands
    plt.close(figure)


def test_comparison_error_maps_name_their_dataset_in_the_title(tmp_path: Path) -> None:
    """The columns of a comparison are models, so the one dataset they share is named in the title; a dataset column is not repeated."""
    frame, _grid = steady_frame(tmp_path)
    behavior = analysis.evaluation.plots.error_behavior
    models = _displayed({"A": frame, "B": _steady_peer(frame)}, dataset="lentil, ID")
    comparison = behavior.plot_std_error_maps(datasets=models, max_cases=_CASE_COUNT)
    assert "· lentil, ID" in comparison.get_suptitle()
    plt.close(comparison)
    single = behavior.plot_std_error_maps(datasets=_displayed({"lentil, ID": frame}), max_cases=_CASE_COUNT)
    assert "lentil, ID" not in single.get_suptitle()
    plt.close(single)


def test_aggregate_titles_name_one_model_or_two_by_their_short_labels_and_more_none(tmp_path: Path) -> None:
    """One model ends the title as its label, two as «A vs B», three none (the legend names them); an unlabelled model gives none."""
    layout = analysis.evaluation.plots.layout
    counts = {"A": 4, "B": 4, "C": 4}
    for models, named in ((("A",), "(A)"), (("A", "B"), "(A vs B)"), (("A", "B", "C"), None), (("",), None)):
        title, _headings = layout.aggregate_title_context("Title", counts, models=models)
        assert title.endswith(named) if named else "(" not in title, models
    labels = {"A · lentil": ("lentil, ID", "A"), "B · lentil": ("lentil, ID", "B"), "A · pea": ("lentil, OOD", "A"), "B · pea": ("lentil, OOD", "B")}
    displayed = dataset_cuts.DisplayFrames(dict.fromkeys(labels, pd.DataFrame()), labels)
    title, _headings = layout.dataset_title_context("Title", displayed, dict.fromkeys(labels, 3))
    assert title.endswith("(A vs B)")  # the models of every dataset panel, each once
    frame, _grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.error_behavior.plot_global_error_metrics(
        datasets=_displayed({"A": frame, "B": _steady_peer(frame)}, dataset="lentil, ID")
    )
    assert any(text.endswith("(A vs B)") for text in _figure_texts(figure))
    plt.close(figure)


def test_the_permeability_overlay_labels_its_signed_error_bar_at_band_centres_with_fixed_decimals(tmp_path: Path) -> None:
    """The overlay's signed-error bar ticks at the band centres (a labelled 0) with one decimal, like the overview's."""
    frame, _grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.samples_outliers.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    figure.canvas.draw()
    levels = _levels(figure.axes[2])  # the error map: first row, third column; its bar is the third slot after the six maps
    np.testing.assert_allclose(figure.axes[8].get_yticks(), (levels[:-1] + levels[1:]) / 2.0)
    labels = [text.get_text() for text in figure.axes[8].get_yticklabels()]
    assert "0.0" in labels or "0.00" in labels
    # Labels thinned for a short bar are blank; the fixed-decimal rule covers the shown ones.
    assert len({len(text.partition(".")[2]) for text in labels if text}) == 1
    # Its bars can show arrows, so one note line under the title says what they mean and which scales run over a percentile range.
    (note,) = (text.get_text() for text in figure.texts if text.get_text().startswith("Colour scales:"))
    assert "; arrows mark values outside; red line: " in note  # the line goes on to name the one red error contour drawn here
    assert "permeability over its full range" in note
    plt.close(figure)


def test_constant_permeability_components_keep_their_one_value_scale_and_a_constant_shear_stays_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A constant Kxx is labelled at its value (log10 mode: its log10).

    A constant shear keeps its value and the signed colormap, a zero shear the neutral centre colour.
    """
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    real_load_case = samples.cases.load_case
    shear = {"value": -2.0e-10}

    def constant_kappa(*args: Any) -> Any:
        case = real_load_case(*args)
        shape = case.coordinates.shape[1:]
        return replace(case, permeability=np.stack([np.full(shape, 1.0e-9), np.full(shape, shear["value"]), np.full(shape, 3.0e-9)]))

    monkeypatch.setattr(samples.cases, "load_case", constant_kappa)
    signed = visual_semantics.field_visual_semantics("Kxy", role="signed_error").colormap
    for kappa_scale, kxx in (("kappa", 1.0e-9), ("log10(kappa)", -9.0)):
        figure = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p", kappa_scale=kappa_scale)
        np.testing.assert_allclose(figure.axes[6].get_yticks(), [kxx])  # the Kxx bar (slots follow the six maps): one tick at its value
        np.testing.assert_allclose(figure.axes[7].get_yticks(), [-2.0e-10])  # the Kxy bar
        assert figure.axes[1].collections[0].get_cmap().name == signed
        plt.close(figure)
    shear["value"] = 0.0  # an isotropic tensor: the zero shear is drawn in the neutral centre colour, never as a positive extreme
    figure = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    mesh = figure.axes[1].collections[0]
    drawn = mesh.to_rgba(np.asarray(mesh.get_array())).reshape(-1, 4)
    np.testing.assert_allclose(drawn, np.broadcast_to(colormaps[signed](0.5), drawn.shape), atol=0.01)
    plt.close(figure)


def test_a_near_uniform_permeability_keeps_distinct_log10_labels_on_state_bands_that_stay_the_log10_of_its_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Kxx varying by 0.02 % is widened like any state: its log10 bar labels stay distinct and are the log10 of the kappa bands."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    real_load_case = samples.cases.load_case

    def near_uniform(*args: Any) -> Any:
        case = real_load_case(*args)
        x, _y = case.coordinates
        kxx = 1.0e-9 * (1.0 + 2.0e-4 * np.sin(3.0 * x))
        return replace(case, permeability=np.stack([kxx, np.zeros_like(kxx), 2.0e-9 * np.ones_like(kxx)]))

    monkeypatch.setattr(samples.cases, "load_case", near_uniform)
    kappa = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p", kappa_scale="kappa")
    log = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
    log.canvas.draw()
    np.testing.assert_allclose(_levels(log.axes[0]), np.log10(_levels(kappa.axes[0])))
    labels = [text.get_text() for text in log.axes[6].get_yticklabels() if text.get_text()]
    assert len(labels) > 1
    assert len(set(labels)) == len(labels)
    plt.close(kappa)
    plt.close(log)


def test_the_permeability_overlay_titles_follow_the_label_style_and_its_red_contours_mark_large_errors_of_either_sign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tensor panels read kappa symbols or names in the active style; the contour levels are quantiles of |error|, here dominated by negatives."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    real_error_values = samples._error_values

    def negative_peak(*args: Any, **kwargs: Any) -> Any:
        result = real_error_values(*args, **kwargs)
        values = 0.01 * np.ones_like(result.map_values)
        values.flat[:2] = -1.0  # two of the cells carry the largest errors, both negative
        return replace(result, map_values=values)

    monkeypatch.setattr(samples, "_error_values", negative_peak)
    titles = {}
    try:
        for style in ("symbol", "name"):
            analysis.presentation.field_labels.set_label_style(style)
            figure = samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p")
            titles[style] = [figure.axes[index].get_title() for index in (0, 1, 3, 4)]  # the tensor panels Kxx, Kxy, Kyx, Kyy
            if style == "symbol":
                red = [cast("Any", item) for item in figure.axes[0].collections if hasattr(item, "levels") and not cast("Any", item).filled]
                assert red
                assert max(red[0].levels) > 0.1  # signed quantiles would sit at 0.01
            plt.close(figure)
    finally:
        analysis.presentation.field_labels.set_label_style("symbol")
    # The headers are bold, which rewrites a mathtext symbol inside its span; the symbols stay recognisable.
    assert [(r"\kappa_{" + suffix + "}") in title for suffix, title in zip(("xx", "xy", "yx", "yy"), titles["symbol"], strict=True)] == [True] * 4
    log10 = r"$\mathbf{\log_{10}}$("  # typeset like the spectral transfer label, set bold with the header
    assert [title.startswith(log10) and title.endswith(" / 1 m²)") for title in titles["symbol"]] == [True, False, False, True]
    assert titles["name"] == [f"{log10}Permeability xx / 1 m²)", "Permeability xy", "Permeability yx", f"{log10}Permeability yy / 1 m²)"]


def test_a_mean_error_map_without_finite_values_names_the_reference_floor_only_in_the_relative_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The below-the-reference-floor reason belongs to the relative mode; every other mode says that no finite values exist."""
    frame, _grid = steady_frame(tmp_path)
    behavior = analysis.evaluation.plots.error_behavior
    full_array = behavior._full_array
    monkeypatch.setattr(behavior, "_full_array", lambda *args: np.full_like(full_array(*args), np.nan))
    texts = {}
    for mode in (error_diagnostics.SIGNED_ERROR, "Relative [%]"):
        figure = behavior.plot_mean_error_maps(datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT, error_mode=mode)
        texts[mode] = {text.get_text() for axis in figure.axes for text in axis.texts if "Unavailable" in text.get_text()}
        plt.close(figure)
    assert texts[error_diagnostics.SIGNED_ERROR] == {"Unavailable: no finite values"}
    assert texts["Relative [%]"]
    assert all("reference floor" in text for text in texts["Relative [%]"])  # the cases it leaves out, or the generic reason
    figure = behavior.plot_mean_error_maps(datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT)
    lines = [text.get_text() for axis in figure.axes for text in axis.texts if text.get_text().startswith("mean ")]
    assert lines
    assert set(lines) == {"mean |e| n/a"}  # an unavailable map keeps its mean-error line, saying so
    plt.close(figure)


def test_the_mean_error_maps_put_the_datasets_in_columns_and_grow_with_the_selection(tmp_path: Path) -> None:
    """A row per field and a column per dataset: each header once on top, equal maps without a width cap, every map with its own bar and unit."""
    frame, _grid = steady_frame(tmp_path)
    plot = analysis.evaluation.plots.error_behavior.plot_mean_error_maps
    one = plot(datasets={"Model A": frame}, max_cases=_CASE_COUNT)
    two = plot(datasets={"Model A": frame, "Model B": _steady_peer(frame)}, max_cases=_CASE_COUNT)
    maps, bars = (np.asarray(axes, dtype=object).reshape(4, 2) for axes in (two.axes[:8], two.axes[8:]))
    assert [axis.get_title() for axis in maps[0]] == ["Model A", "Model B"]
    assert not any(axis.get_title() for axis in maps[1:].flat)
    assert [bar.get_ylabel() for bar in bars.flat] == ["[Pa]", "[Pa]", *["[m/s]"] * 6]
    assert two.get_figwidth() > one.get_figwidth()
    for figure in (one, two):
        figure.canvas.draw()
    widths = [figure.axes[0].get_position().width * figure.get_figwidth() for figure in (one, two)]
    assert widths[0] == pytest.approx(widths[1])  # the second column adds a cell, the maps keep their size
    # One mean-error line above every map, computed from that map in the unit of its bar; the standard deviation maps have none.
    for map_axis, bar in zip(maps.flat, bars.flat, strict=True):
        (line,) = (text.get_text() for text in map_axis.texts if text.get_text().startswith("mean "))
        drawn = np.ma.filled(map_axis.collections[0].get_array().astype(float), np.nan)  # the fixture grid is drawn unsampled
        mean = analysis.evaluation.plots.layout.significant(float(np.nanmean(np.abs(drawn))))
        assert line == f"mean |e| = {mean} {bar.get_ylabel().strip('[]')}"
    plt.close(one)
    plt.close(two)
    deviation = analysis.evaluation.plots.error_behavior.plot_std_error_maps(datasets={"Model A": frame}, max_cases=_CASE_COUNT)
    assert not [text for axis in deviation.axes for text in axis.texts if text.get_text().startswith("mean ")]
    plt.close(deviation)


def test_the_permeability_overlay_writes_the_kappa_bands_as_log10_values_and_keeps_the_shear_components_signed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kxx and Kyy have eleven geometric bands; log10 mode writes them as log10 values on a unit-free bar; Kxy and Kyx are symmetric round bands."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    real_load_case = samples.cases.load_case

    def varying_kappa(*args: Any) -> Any:
        case = real_load_case(*args)
        x, y = case.coordinates
        kxx, kxy, kyy = 1.0e-9 * np.exp(np.sin(3.0 * x) * np.cos(4.0 * y)), 3.0e-10 * np.sin(2.0 * x - 3.0 * y), 2.0e-9 * (1.5 + np.cos(2.0 * x + y))
        return replace(case, permeability=np.stack([kxx, kxy, kyy]))

    monkeypatch.setattr(samples.cases, "load_case", varying_kappa)
    figures = {
        scale: samples.plot_permeability_error_overlay(datasets={"lentil, ID": frame}, row_position=0, channel="p", kappa_scale=scale)
        for scale in ("kappa", "log10(kappa)")
    }
    kappa, log = figures["kappa"], figures["log10(kappa)"]
    for index in (0, 4):  # Kxx and Kyy; the colour-bar slots follow the six maps
        assert _levels(kappa.axes[index]).size == 12
        np.testing.assert_allclose(_levels(log.axes[index]), np.log10(_levels(kappa.axes[index])))
        assert (log.axes[6 + index].get_ylabel(), kappa.axes[6 + index].get_ylabel().endswith("m²]")) == ("[-]", True)
    for index in (1, 3):  # Kxy and Kyx are not written as logarithms in either mode
        for figure in figures.values():
            shear = _levels(figure.axes[index])
            assert _round_steps(shear)
            assert shear[0] == pytest.approx(-shear[-1])
    plt.close(kappa)
    plt.close(log)


def test_the_brinkman_residual_puts_its_cdfs_in_a_band_above_the_maps_on_one_locked_scale(tmp_path: Path) -> None:
    """One residual map per dataset on one scale with its own bar, the CDF panel above them, its legend inside, in architecture colours."""
    frame, _grid = steady_frame(tmp_path)
    other = _steady_peer(frame)
    other.attrs["artifact_provenance"]["model"].update(kind="uno")
    datasets = {"Model A": frame, "Model B": other}
    figure = analysis.evaluation.plots.physical_consistency.plot_brinkman_residual(datasets=datasets, max_cases=_CASE_COUNT)
    figure.canvas.draw()
    maps, bars, cdf = figure.axes[:2], figure.axes[2:4], figure.axes[4]
    assert cdf.get_position().y0 > max(axis.get_position().y1 for axis in maps)
    assert cdf.get_legend() is not None
    assert not figure.legends
    np.testing.assert_array_equal(_levels(maps[0]), _levels(maps[1]))
    assert all(bar.get_ylabel().endswith("[-]") for bar in bars)
    assert [axis.get_title() for axis in maps] == ["Model A", "Model B"]
    architectures = [visual_semantics.architecture_color(family) for family in ("fno", "uno")]
    assert [line.get_color() for line in cdf.lines] == architectures
    plt.close(figure)


def test_the_spatial_residual_profile_shares_the_y_range_of_the_maps(tmp_path: Path) -> None:
    """The maps of all datasets stand side by side and one profile panel of all of them right of them, as tall as the maps and on their y range."""
    frame, _grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.physical_consistency.plot_spatial_residuals(datasets={"Model A": frame, "Model B": _steady_peer(frame)})
    figure.canvas.draw()
    maps, profile = figure.axes[:2], figure.axes[-1]
    assert profile.get_ylim() == maps[0].get_ylim()
    for axis in (*maps, profile):
        assert axis.get_position().y0 == pytest.approx(maps[0].get_position().y0)
        assert axis.get_position().y1 == pytest.approx(maps[0].get_position().y1)
    assert profile.get_position().x0 > max(axis.get_position().x1 for axis in figure.axes[2:4])  # right of the last colour bar
    assert len(profile.lines) == len(maps) == 2
    assert profile.get_legend() is not None
    assert all(bar.get_ylabel().endswith("[-]") for bar in figure.axes[2:4])
    plt.close(figure)


def test_the_physical_consistency_figures_draw_a_row_per_dataset_and_the_models_as_lines_with_one_legend_beside_them(tmp_path: Path) -> None:
    """Rule 17: a row of panels per dataset, the models inside it in their series colours, one legend outside and no legend axis."""
    shown = _two_models_on_two_datasets(tmp_path)
    styles = analysis.evaluation.plots.series.series_styles(shown)
    physical = analysis.evaluation.plots.physical_consistency
    for plot, columns in ((physical.plot_flux_balance, 2), (physical.plot_pressure_consistency, 2), (physical.plot_physical_consistency_cdf_grid, 4)):
        figure = plot(datasets=shown, max_cases=_CASE_COUNT)
        assert len(figure.axes) == 2 * columns
        for row, dataset in enumerate(("ID", "OOD")):
            for axis in figure.axes[row * columns : (row + 1) * columns]:
                assert [line.get_color() for line in axis.lines] == [styles[f"{model} · {dataset}"].color for model in ("A", "B")]
        assert {legend.get_title().get_text() for legend in figure.legends} == {"Model", "Role"}
        plt.close(figure)
    single = physical.plot_flux_balance(datasets=_displayed({"Model A": next(iter(shown.values()))}), max_cases=_CASE_COUNT)
    assert len(single.axes) == 2  # one model of one dataset: one panel set and no legend
    assert not single.legends
    plt.close(single)


def test_the_pressure_boundary_summary_puts_the_dimensionless_medians_and_the_pascal_error_on_two_panels(tmp_path: Path) -> None:
    """
    [-] medians (inlet, outlet, boundary) and the drop error in Pa never share an axis; each model keeps its series colour in both.

    The medians are points, never bars, whose length on a log axis would suggest a baseline; the legend names the models by
    marker entries, and the row headers name each dataset and so its role.
    """
    shown = _two_models_on_two_datasets(tmp_path)
    physical = analysis.evaluation.plots.physical_consistency
    table = physical.build_pressure_boundary_summary(shown)
    styles = analysis.evaluation.plots.series.series_styles(shown)
    figure = physical.plot_pressure_boundary_summary(datasets=shown)
    assert len(figure.axes) == 2 * 2  # a row per dataset, the two measures side by side
    for row, dataset in enumerate(("ID", "OOD")):
        keys = [f"{model} · {dataset}" for model in ("A", "B")]
        dimensionless, pascal = figure.axes[2 * row : 2 * row + 2]
        assert not dimensionless.patches
        assert not pascal.patches
        assert dimensionless.get_ylabel().endswith("[-]")
        assert pascal.get_ylabel().endswith("[Pa]")
        assert all(line.get_linestyle() == "None" and line.get_marker() == "o" for line in (*dimensionless.lines, *pascal.lines))
        np.testing.assert_allclose([_numeric_array(line.get_ydata()) for line in dimensionless.lines], table.loc[keys].iloc[:, :3].to_numpy())
        np.testing.assert_allclose([_numeric_array(line.get_ydata()) for line in pascal.lines], table.loc[keys].iloc[:, 3:].to_numpy())
        assert [to_rgb(line.get_color()) for line in pascal.lines] == [to_rgb(styles[key].color) for key in keys]
    (legend,) = figure.legends
    assert legend.get_title().get_text() == "Model"
    assert all(cast("Any", handle).get_marker() == "o" for handle in legend.legend_handles)
    plt.close(figure)


def test_the_spectral_figures_draw_all_channels_in_one_colour_each_with_a_panel_per_dataset_and_a_row_per_model(tmp_path: Path) -> None:
    """Datasets are columns, models rows; a channel keeps its colour in every curve kind and panel; the y label stands once per row."""
    shown = _two_models_on_two_datasets(tmp_path)
    fields = analysis.evaluation.presentation.shared_display_fields(tuple(shown.values()))
    spectral = analysis.evaluation.plots.spectral_fidelity
    demand = spectral.plot_spectral_demand_prediction_error(datasets=shown, max_cases=_CASE_COUNT)
    assert len(demand.axes) == 2 * 2 * 2  # spectrum and error row per model, a column per dataset
    channel_colours = [line.get_color() for line in _medians(demand.axes[0])[::2]]
    assert len(set(channel_colours)) == len(fields) > 1  # every channel, each its own colour
    for axis in (*demand.axes[0:2], *demand.axes[4:6]):  # spectrum panels: reference (dashed) and prediction (solid) of every channel
        assert [line.get_color() for line in _medians(axis)] == [colour for colour in channel_colours for _curve in range(2)]
        assert [line.get_linestyle() for line in _medians(axis)[:2]] == ["--", "-"]
    for axis in (*demand.axes[2:4], *demand.axes[6:8]):  # error panels
        assert [line.get_color() for line in _medians(axis)] == channel_colours
    for axis in demand.axes:  # a band edge has its curve's colour, like every line of the panel
        assert {line.get_color() for line in axis.lines} == set(channel_colours)
    labels = [axis.get_ylabel() for axis in demand.axes]
    assert [bool(label) for label in labels] == [True, False, True, False] * 2  # once per row
    assert all(label.endswith("[-]") for label in labels if label)  # normalised spectra: no unit mixes on one axis
    legends = {legend.get_title().get_text(): legend for legend in demand.legends}
    assert [text.get_text() for text in legends["Channels"].get_texts()] == [field.matplotlib_label for field in fields]
    plt.close(demand)
    transfer = spectral.plot_spectral_transfer_ratio(datasets=shown, max_cases=_CASE_COUNT)
    assert len(transfer.axes) == 2 * 2 * 2  # transfer and support row per model
    assert [line.get_color() for line in _medians(transfer.axes[0])[: len(fields)]] == channel_colours
    assert [bool(axis.get_ylabel()) for axis in transfer.axes[:4]] == [True, False, True, False]
    plt.close(transfer)


def test_a_residual_cdf_names_the_values_a_log_axis_cannot_show(tmp_path: Path) -> None:
    """A model with only zero residuals draws no line but says why; one with some zeros names how many of its values are left off."""
    frame, _grid = steady_frame(tmp_path)
    frame["flux_balance_mse"] = 0.0
    partial = _steady_peer(frame)
    partial["flux_balance_mse"] = [0.0, 0.04]
    datasets = _displayed({"Model A": frame, "Model B": partial}, dataset="ID")
    figure = analysis.evaluation.plots.physical_consistency.plot_flux_balance(datasets=datasets, max_cases=_CASE_COUNT)
    notes = [text.get_text() for text in figure.axes[0].texts if "not shown" in text.get_text()]
    assert notes == [f"Model A: {_CASE_COUNT} of {_CASE_COUNT} values ≤ 0 not shown", f"Model B: 1 of {_CASE_COUNT} values ≤ 0 not shown"]
    assert len(figure.axes[0].lines) == 1  # only model B has a positive value to draw
    plt.close(figure)


def test_a_spectral_column_is_titled_on_its_first_visible_panel_when_the_first_model_lacks_the_dataset(tmp_path: Path) -> None:
    """Model A has no OOD frame, so its OOD panels are hidden; the OOD column still carries its dataset name."""
    shown = _two_models_on_two_datasets(tmp_path)
    keys = [key for key in shown if key != "A · OOD"]
    partial = dataset_cuts.DisplayFrames({key: shown[key] for key in keys}, {key: shown.labels[key] for key in keys})
    spectral = analysis.evaluation.plots.spectral_fidelity
    for plot in (spectral.plot_spectral_demand_prediction_error, spectral.plot_spectral_transfer_ratio):
        figure = plot(datasets=partial, max_cases=_CASE_COUNT)
        assert [axis.get_title() for axis in figure.axes if axis.get_visible() and axis.get_title()] == ["ID", "OOD"]
        plt.close(figure)


def test_the_pressure_boundary_medians_leave_a_non_finite_case_out_and_name_a_model_without_a_finite_one(tmp_path: Path) -> None:
    """One NaN case still gives a finite median and a bar; a model whose every case is NaN gets no bar but an unavailable note."""
    frame, _grid = steady_frame(tmp_path)
    frame.loc[0, "pressure_inlet_mse"] = np.nan
    empty = _steady_peer(frame)
    empty[["pressure_inlet_mse", "pressure_outlet_mean_square", "pressure_boundary_mse"]] = np.nan
    datasets = _displayed({"Model A": frame, "Model B": empty}, dataset="ID")
    physical = analysis.evaluation.plots.physical_consistency
    table = physical.build_pressure_boundary_summary(datasets)
    inlet = table.columns[0]
    assert table.loc["Model A", inlet] == pytest.approx(frame["pressure_inlet_mse"].iloc[1])
    assert pd.isna(table.loc["Model B", inlet])
    figure = physical.plot_pressure_boundary_summary(datasets=datasets)
    assert np.isfinite(_numeric_array(figure.axes[0].lines[0].get_ydata())[0])
    assert figure.axes[0].get_yscale() == "log"  # the finite medians are positive; the missing one no longer switches the scale off
    assert any("unavailable" in text.get_text() for text in figure.axes[0].texts)
    plt.close(figure)


def test_the_overview_bars_of_near_uniform_state_maps_keep_three_decimals_and_no_power_of_ten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state field of 26.85 ± 5e-5 is widened to bands of 0.001: its reference and prediction bars read three decimals and no factor."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    real_case_field = samples.presentation.case_field

    def near_uniform(*args: Any, **kwargs: Any) -> np.ndarray:
        values = np.asarray(real_case_field(*args, **kwargs), dtype=float)
        return 26.85 + 1e-4 * ((values - values.min()) / max(np.ptp(values), 1e-12) - 0.5)

    monkeypatch.setattr(samples.presentation, "case_field", near_uniform)
    figure = samples.plot_sample_prediction_overview(datasets={"lentil, ID": frame}, row_position=0)
    figure.canvas.draw()
    slots = figure.axes[len(figure.axes) // 2 :]  # the colour-bar slots, row-major over reference, prediction, error
    for slot in slots[0::3] + slots[1::3]:
        assert "10" not in slot.get_ylabel()
        assert {len(text.get_text().partition(".")[2]) for text in slot.get_yticklabels() if text.get_text()} == {3}
    plt.close(figure)


def test_the_overview_can_show_the_velocity_magnitude_with_streamlines_in_place_of_its_components(tmp_path: Path) -> None:
    """fields=("p", "U") draws two rows; the U row shows the reference speed with the pair's streamlines and its own mean line."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    figure = samples.plot_sample_prediction_overview(datasets={"lentil, ID": frame}, row_position=0, fields=("p", "U"))
    maps = figure.axes[:6]
    assert len(figure.axes) == 12  # two rows of three maps and their colour-bar slots
    speed = samples.presentation.case_field(samples.cases.load_case(frame, 0), samples._resolve_display_field(frame, "U"), source="reference")
    drawn = np.ma.filled(cast("Any", maps[3].collections[0]).get_array().astype(float), np.nan)
    assert np.nanmax(drawn) == pytest.approx(float(np.max(speed)))
    assert any(isinstance(collection, LineCollection) for collection in maps[3].collections[1:])  # streamlines over the speed
    labels = [text.get_text() for axis in maps for text in axis.texts if text.get_gid() == "channel-row-label"]
    assert any(r"|\mathbf{u}|" in label for label in labels)  # the row label |u|, set bold like every row label
    assert next(text.get_text() for text in maps[5].texts if text.get_gid() != "channel-row-label").startswith("mean |e| = ")
    plt.close(figure)


def test_the_case_titles_end_with_the_short_labels_of_the_compared_models(tmp_path: Path) -> None:
    """One compared model ends the sample and kappa titles as its label, two the p/U title as «A vs B», more none; a bare mapping names none."""
    frame, _grid = steady_frame(tmp_path)
    samples = analysis.evaluation.plots.samples_outliers
    shown = dataset_cuts.DisplayFrames({"lentil, ID": frame}, {"lentil, ID": ("lentil, ID", "FNO · B · seed 9")})
    for plot in (samples.plot_sample_prediction_overview, samples.plot_permeability_error_overlay):
        figure = plot(datasets=shown, row_position=0)
        assert figure.get_suptitle().endswith(" (FNO · B · seed 9)"), plot.__name__
        plt.close(figure)
        bare = plot(datasets={"lentil, ID": frame}, row_position=0)
        assert "(" not in bare.get_suptitle(), plot.__name__
        plt.close(bare)
    models = _displayed({"FNO · B · seed 9": frame, "U-NO · B · seed 9": _steady_peer(frame), "RNO · B · seed 9": _steady_peer(frame)}, dataset="ID")
    for compared, suffix in (
        (("FNO · B · seed 9", "U-NO · B · seed 9"), " (FNO · B · seed 9 vs U-NO · B · seed 9)"),
        (("U-NO · B · seed 9",), " (U-NO · B · seed 9)"),
    ):
        figure = samples.plot_pressure_velocity_comparison(datasets=models, models=compared)
        assert figure.get_suptitle() == f"Pressure and velocity comparison · Lentil, ID — Case 1{suffix}"
        plt.close(figure)
    three = samples.plot_pressure_velocity_comparison(datasets=models)
    assert three.get_suptitle() == "Pressure and velocity comparison · Lentil, ID — Case 1"  # the column headers name the three models
    plt.close(three)


def test_the_mean_error_maps_show_the_signed_mean_on_a_symmetric_blue_white_red_scale_by_default(tmp_path: Path) -> None:
    """The default mean error map is the mean of prediction minus reference over the selected cases, symmetric about 0."""
    frame, x_grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.error_behavior.plot_mean_error_maps(datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT)
    assert "Mean signed error maps" in figure.get_suptitle()
    levels = _levels(figure.axes[0])
    assert levels[0] == pytest.approx(-levels[-1])
    assert figure.axes[0].collections[0].get_cmap().name == visual_semantics.field_visual_semantics("error", role="signed_error").colormap
    assert levels[-1] >= np.percentile(0.75 * x_grid, 95.0)  # the fixture predicts above the reference everywhere: the mean is 0.75 x
    figure.canvas.draw()
    bars = figure.axes[len(figure.axes) // 2 :]  # the colour-bar slots, one per map
    labels = [text.get_text() for text in bars[0].get_yticklabels()]  # the signed bar labels its band centres at fixed decimals
    assert "0.0" in labels or "0.00" in labels
    assert len({len(text.partition(".")[2]) for text in labels}) == 1
    plt.close(figure)
    frame.attrs["artifact_provenance"]["normalizer"].update(normalization=domain.tasks.spec.TRAIN_SCALING_KIND, fit_split="train", sha256="f" * 64)
    for mode in ("MAE", error_diagnostics.NORMALIZED_ABSOLUTE_ERROR):  # the other modes tick on round boundaries from 0
        other = analysis.evaluation.plots.error_behavior.plot_mean_error_maps(
            datasets={"Steady fixture": frame}, max_cases=_CASE_COUNT, error_mode=mode
        )
        np.testing.assert_allclose(other.axes[len(other.axes) // 2].get_yticks()[0], 0.0, atol=1e-12)
        plt.close(other)


def test_the_mae_overview_heads_its_error_column_absolute_error_on_round_levels_from_zero_in_difference_units(tmp_path: Path) -> None:
    """The MAE column is "Absolute error"; its bars carry the fields' difference units and start at 0."""
    frame, _grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview(datasets={"lentil, ID": frame}, error_mode="MAE")
    errors, bars = figure.axes[2:9:3], figure.axes[11::3]
    assert errors[0].get_title() == "Absolute error"
    assert [bar.get_ylabel() for bar in bars] == ["[Pa]", "[m/s]", "[m/s]"]
    # |e| of p is 0.5 x over x in [0, 2] m: its 95th percentile of 0.95 Pa is reached by ten bands of 0.1 Pa.
    np.testing.assert_allclose(_levels(errors[0]), np.linspace(0.0, 1.0, 11))
    plt.close(figure)


def test_a_nonfinite_prediction_overview_masks_and_notes_its_cells_and_takes_the_reference_levels(tmp_path: Path) -> None:
    """The pressure prediction's NaN cell stays masked with the note; it draws on the reference's levels; its error states why it is missing."""
    frame = _nonfinite_frame(tmp_path / "nonfinite", (0,))
    figure = analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview(datasets={"lentil, ID": frame}, error_mode="MAE")
    prediction = figure.axes[1]
    drawn = cast("Any", prediction.collections[0]).get_array()
    assert drawn is not None
    assert int(np.ma.getmaskarray(drawn).sum()) == 1
    assert [text.get_text() for text in prediction.texts] == ["Non-finite cells masked"]
    np.testing.assert_array_equal(_levels(prediction), _levels(figure.axes[0]))
    assert _levels(prediction)[[0, -1]] == pytest.approx((0.0, 1.0))
    assert figure.axes[2].texts[0].get_text().startswith("Unavailable: Non-finite")
    plt.close(figure)


def test_the_steady_work_table_has_one_row_per_run(tmp_path: Path) -> None:
    """Two labelled frames of one run give one row, named by the first; another run keeps its own row."""
    frame, _grid = steady_frame(tmp_path)
    frames = {"ID": frame, "Material OOD": frame.copy(), "Other run": frame.copy()}
    for copied in ("Material OOD", "Other run"):
        frames[copied].attrs = copy.deepcopy(frame.attrs)
    frames["Other run"].attrs[dataframe.COMPLETED_RUN_SUMMARY_ATTR]["run_name"] = "steady-other"
    run_summary = analysis.evaluation.plots.run_summary

    table = run_summary.build_frame_work_table(frames, run_summary._steady_inference_seconds)

    assert list(table.index) == ["ID", "Other run"]
    assert list(table["run_name"]) == ["steady-fixture", "steady-other"]
    assert table.loc["ID", "inference_seconds_per_case"] == pytest.approx(run_summary._steady_inference_seconds("ID", frame))


def _break_even_entry(frames: Mapping[str, pd.DataFrame], *, comparison: bool) -> Any:
    """Return the factory of the runtime section's Break-even entry over ``frames``."""
    registry = panel._build_sections(_scope(frames, comparison=comparison), comparison=comparison)
    return {title: factory for title, factory, _stem in registry["runtime"][0]}["Break-even"]


def test_the_steady_break_even_entry_costs_each_run_and_names_a_run_without_training_evidence(tmp_path: Path) -> None:
    """A run with training seconds and train cases gets a panel from its paired timing; one without is named below the figure."""
    frame, _grid = steady_frame(tmp_path)
    frames = _model_copies(frame, ("Run A", "Run B"))
    frames["Run B"].attrs[dataframe.COMPLETED_RUN_SUMMARY_ATTR]["run_name"] = "steady-other"
    frames["Run A"].attrs[dataframe.COMPLETED_RUN_SUMMARY_ATTR].update(model_kind="fno", elapsed_seconds=3600.0, training_units={"train_cases": 10})

    figure, _details = _break_even_entry(frames, comparison=True)()
    try:
        assert [axis.get_title() for axis in figure.axes if axis.lines] == [""]  # one panel: the figure title and key name its run
        assert any("Run A" in artist.get_text() for artist in figure.findobj(lambda artist: hasattr(artist, "get_text")))
        assert "Run B" in figure.get_supxlabel()
        # The costs use the clocks and the paired cases of the speed-up view: its solver sums over its one paired case.
        speedup = analysis.evaluation.plots.run_summary.build_runtime_table({"Run A": frames["Run A"]}).iloc[0]
        assert (speedup["solver_speedup Σ comsol_solver_s"], speedup["solver_speedup Σ surrogate forward [s]"]) == (4.0, 0.1)
        paired, unpaired = speedup["solver_speedup paired cases"], speedup["solver_speedup unavailable cases"]
        # Three significant digits with their trailing zeros, written out independently of the formatter.
        expected = rf"$t_\mathrm{{c}}$ = 4.00 s, $t_\mathrm{{s}}$ = 0.100 s per case over {paired} paired cases ({unpaired} unpaired)"
        assert expected in "\n".join(_figure_texts(figure))
    finally:
        plt.close("all")


def test_the_steady_break_even_entry_states_that_no_run_has_the_evidence(tmp_path: Path) -> None:
    """Without any run's training evidence the entry shows a message naming the runs, not a failing view."""
    frame, _grid = steady_frame(tmp_path)

    shown = _break_even_entry({"Steady fixture": frame}, comparison=False)()

    assert isinstance(shown, widgets.HTML)
    assert "Steady fixture" in shown.value


def test_every_steady_registry_entry_builds_with_steady_fields_only(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Invoke each visible steady entry once, with no view failing, and expose only steady output fields."""
    frame, _x_grid = steady_frame(tmp_path)
    with analysis.evaluation.session.EvaluationSession({"Steady fixture": frame}):  # bound like the workflow's frames, so cuts bind beside it
        registry = panel._build_sections(_scope({"Steady fixture": frame}, comparison=False), comparison=False)
        entries = [entry for section_entries, _section_title in registry.values() for entry in section_entries]

        assert entries
        assert len({stem for _title, _factory, stem in entries}) == len(entries)
        try:
            for title, factory, stem in entries:
                analysis.ui.notebook._invoke_dropdown_entry(
                    factory,
                    export_state={"fig": None, "plot_name": stem, "title": title},
                    export_plot_name=stem,
                    export_title=title,
                )
        finally:
            plt.close("all")
    assert not [record.getMessage() for record in caplog.records if "View failed" in record.getMessage()]
    assert {field.key for field in analysis.evaluation.presentation.display_fields(frame)} == {"p", "u", "v", "velocity_magnitude"}


def _model_copies(frame: pd.DataFrame, labels: tuple[str, ...], *, role: str = "eval") -> dict[str, pd.DataFrame]:
    """Return independent copies of ``frame`` as models of one split role."""
    copies = {label: frame.copy(deep=True) for label in labels}
    for copied in copies.values():
        copied.attrs = copy.deepcopy(frame.attrs)
        copied.attrs["dataset_role"] = role
    return copies


def _pressure_velocity_view(
    scope: Any, monkeypatch: pytest.MonkeyPatch, export_state: dict[str, Any] | None = None
) -> tuple[Any, list[dict[str, Any]]]:
    """Build the comparison's pressure-velocity view with its plot calls recorded."""
    drawn: list[dict[str, Any]] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda _view, _plot, **kwargs: drawn.append(kwargs["kwargs"]) or True)
    registry = panel._build_sections(scope, comparison=True)
    (factory,) = [
        factory for entries, _title in registry.values() for title, factory, _stem in entries if title == "Pressure & velocity field comparison"
    ]
    return analysis.ui.notebook._invoke_dropdown_entry(factory, export_state=export_state, export_plot_name="pv", export_title="pv"), drawn


def test_an_unsupported_family_is_not_preselected_in_the_run_choice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A model whose family the selector cannot offer stays disabled and the defaults are the first two selectable models."""
    frame, _x_grid = steady_frame(tmp_path)
    models = _model_copies(frame, ("A", "B", "C"))
    models["A"].attrs["artifact_provenance"]["model"].update(kind="rno", physics_enabled=True)  # pi-rno is not an architecture
    _view, drawn = _pressure_velocity_view(_scope(models, comparison=True), monkeypatch)
    assert drawn[-1]["models"] == ("B", "C")


def test_the_export_stem_names_the_selected_models_and_a_dropped_model_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stem follows the models that reach the plot; a selected model the drawn role lacks is named on the status line."""
    frame, _x_grid = steady_frame(tmp_path)
    export_state: dict[str, Any] = {"fig": None, "plot_name": "pv", "title": "pv"}
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets=_model_copies(frame, ("A", "B", "C"))),
        panel.EvaluationContext(key="material_ood", label="Material OOD", datasets=_model_copies(frame, ("A", "B"), role="material_ood")),
    ]
    scope = panel._SteadyScope(contexts, comparison=True, refused_roles={})
    view, drawn = _pressure_velocity_view(scope, monkeypatch, export_state)
    boxes = {item.description: item for item in _widget_tree(view.widget) if isinstance(item, widgets.Checkbox)}
    boxes["A"].value = False
    boxes["C"].value = True
    assert drawn[-1]["models"] == ("B", "C")
    assert "B-C" in str(export_state["filename_stem"]).split("__")
    other = [item for item in _widget_tree(view.widget) if isinstance(item, widgets.ToggleButton) and "OOD" in item.description and not item.value]
    other[0].value = True
    assert drawn[-1]["models"] == ("B",)
    assert view.status.text.startswith("C:")


def test_a_role_switch_passes_only_the_models_of_the_drawn_role(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After the role changes the drawn models, no model of the other role reaches the plot."""
    frame, _x_grid = steady_frame(tmp_path)
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets=_model_copies(frame, ("A", "B"))),
        panel.EvaluationContext(key="material_ood", label="Material OOD", datasets=_model_copies(frame, ("C", "D"), role="material_ood")),
    ]
    scope = panel._SteadyScope(contexts, comparison=True, refused_roles={})
    view, drawn = _pressure_velocity_view(scope, monkeypatch)
    assert drawn[-1]["models"] == ("A", "B")
    other = [
        item
        for item in _widget_tree(view.widget)
        if isinstance(item, widgets.ToggleButton) and "OOD" in item.description and not item.disabled and not item.value
    ]
    assert other
    other[0].value = True
    assert set(drawn[-1]["models"]) == {"C", "D"}


def test_the_pressure_velocity_view_offers_its_models_as_runs_and_draws_inside_the_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The runs disclosure names «k of n selected» with the first two selected; the selection reaches the plot as ``models``."""
    frame, _x_grid = steady_frame(tmp_path)
    models = {label: frame.copy(deep=True) for label in ("A", "B", "C")}
    for copied in models.values():
        copied.attrs = copy.deepcopy(frame.attrs)
    drawn: list[dict[str, Any]] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda _view, _plot, **kwargs: drawn.append(kwargs["kwargs"]) or True)
    registry = panel._build_sections(_scope(models, comparison=True), comparison=True)
    (factory,) = [
        factory for entries, _title in registry.values() for title, factory, _stem in entries if title == "Pressure & velocity field comparison"
    ]

    view = analysis.ui.notebook._invoke_dropdown_entry(factory, export_state=None, export_plot_name="pv", export_title="pv")

    summaries = [str(getattr(item, "summary", "")) for item in _widget_tree(view.widget)]
    assert "Runs: 2 of 3 selected" in summaries
    assert drawn[-1]["models"] == ("A", "B")
    assert not {"model_1", "model_2"} & set(drawn[-1])
    figure = analysis.evaluation.plots.samples_outliers.plot_pressure_velocity_comparison(datasets=models, models=("A", "B", "C"))
    figure.canvas.draw()
    title = figure.get_suptitle()
    assert title
    title_box = cast("Any", figure)._suptitle.get_window_extent()
    assert figure.bbox.x0 <= title_box.x0 <= title_box.x1 <= figure.bbox.x1
    assert figure.bbox.y0 <= title_box.y0 <= title_box.y1 <= figure.bbox.y1
    plt.close(figure)


@pytest.mark.parametrize(
    ("comparison", "pooling"), [(False, set()), (True, {"Global comparison summary", "Exact trainable parameters vs performance"})]
)
def test_the_pooling_switch_is_enabled_only_in_the_views_that_weight_every_material_equally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, comparison: bool, pooling: set[str]
) -> None:
    """A comparison pools in its scoreboard and capacity view, a single-model panel has neither; every other view disables the switch."""
    frame, _x_grid = steady_frame(tmp_path)
    models = {"A": frame, "B": _steady_peer(frame)} if comparison else {"Steady fixture": frame}
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)
    registry = panel._build_sections(_scope(models, comparison=comparison), comparison=comparison)
    enabled: dict[str, bool] = {}
    for entries, _section_title in registry.values():
        for title, factory, _stem in entries:
            view = analysis.ui.notebook._invoke_dropdown_entry(factory, export_state=None, export_plot_name=title, export_title=title)
            if isinstance(view, analysis.ui.panel.ViewPanel):
                (toggle,) = [item for item in _widget_tree(view.widget) if getattr(item, "description", None) == analysis.ui.selectors.POOL_LABEL]
                enabled[title] = not toggle.disabled
                assert toggle.disabled == (toggle.tooltip == dataset_cuts.POOL_DISABLED_REASON)
    assert "Summary table" in enabled
    assert {title for title, on in enabled.items() if on} == pooling
    assert len(enabled) > 5


def _pooled_frames(tmp_path: Path) -> tuple[dataset_cuts.DisplayFrames, float]:
    """Return what models A and B draw for the pooled ID role of lentil (first case) and chickpea (second), and the role's whole objective."""
    frame, _x_grid = steady_frame(tmp_path)
    frame.loc[1, "material_family"] = "chickpea"
    scope = _scope({"A": frame, "B": _steady_peer(frame)}, comparison=True)
    choices = selection.resolve_choices(scope.inventory, [item.key for item in scope.inventory.items], pooled=True)
    return scope.frames(choices), frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"]


def test_a_pooled_role_draws_one_frame_per_model_and_material_and_never_one_of_appended_rows(tmp_path: Path) -> None:
    """In a comparison and for one model alike: every frame holds one material, all under the one pooled dataset label."""
    frame, _x_grid = steady_frame(tmp_path)
    frame.loc[1, "material_family"] = "chickpea"
    for models in ({"A": frame, "B": _steady_peer(frame)}, {"Steady fixture": frame}):
        scope = _scope(models, comparison=len(models) > 1)

        drawn = scope.frames(selection.resolve_choices(scope.inventory, [item.key for item in scope.inventory.items], pooled=True))

        assert len(drawn) == 2 * len(models)
        assert [set(cut["material_family"]) for cut in drawn.values()] == [{"chickpea"}, {"lentil"}] * len(models)
        assert {dataset for dataset, _model in drawn.labels.values()} == {"ID, all materials"}
        grouped = analysis.evaluation.plots.series.panel_series(drawn.labels)
        assert [len(keys) for by_model in grouped.values() for keys in by_model.values()] == [2] * len(models)


def test_capacity_shows_the_mean_of_the_material_objectives_with_each_material_as_a_marker(tmp_path: Path) -> None:
    """A model's point is the unweighted mean of its materials' objectives, not the objective of the pooled rows; the materials stay as markers."""
    drawn, whole_role = _pooled_frames(tmp_path)
    objective = {key: frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"] for key, frame in drawn.items()}
    assert objective["A · Chickpea, ID"] != pytest.approx(objective["A · Lentil, ID"])

    figure = analysis.evaluation.plots.sensitivity_capacity.plot_capacity_accuracy(datasets=drawn, max_cases=_CASE_COUNT)

    points = [_numeric_array(collection.get_offsets())[0] for collection in figure.axes[0].collections]
    assert len(points) == 6  # per model its pooled point, then chickpea and lentil
    for first, (model, trainable) in zip((0, 3), (("A", 8.0), ("B", 16.0)), strict=True):
        chickpea, lentil = objective[f"{model} · Chickpea, ID"], objective[f"{model} · Lentil, ID"]
        expected = np.asarray([(trainable, (chickpea + lentil) / 2), (trainable, chickpea), (trainable, lentil)])
        np.testing.assert_allclose(points[first : first + 3], expected)
        assert points[first][1] != pytest.approx(whole_role)  # the objective of the rows appended is another number
    texts = _figure_texts(figure)
    assert {"Chickpea", "Lentil"} <= set(texts)
    assert any(text.startswith("Pooled series") for text in texts)
    plt.close(figure)


def test_the_scoreboard_pools_the_materials_as_the_mean_ratio_to_the_best_with_the_materials_as_markers(tmp_path: Path) -> None:
    """Every bar is the mean of its model's two material markers on the same ratio scale, whose best is the best pooled value."""
    drawn, _whole_role = _pooled_frames(tmp_path)
    for key, factor in (("B · Chickpea, ID", 3.0), ("B · Lentil, ID", 2.0)):  # make model B worse by a different factor per material
        aggregate = drawn[key].attrs[dataframe.PRIMARY_OBJECTIVE_ID]
        aggregate["value"] *= factor
        for statistics in aggregate["group_statistics"].values():
            statistics["normalized_rmse"] *= factor
    raw = {key: frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"] for key, frame in drawn.items()}
    pooled = {model: (raw[f"{model} · Chickpea, ID"] + raw[f"{model} · Lentil, ID"]) / 2 for model in ("A", "B")}

    figure = analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(datasets=drawn)

    axis = figure.axes[0]
    columns = len(axis.get_xticklabels())
    bars = [cast("Rectangle", patch) for patch in axis.patches]
    heights = [bar.get_height() for bar in bars]
    assert len(heights) == 2 * columns  # one bar per model and diagnostic, none per material
    assert heights[0] == pytest.approx(1.0)  # task macro: model A is the best pooled value
    assert heights[columns] == pytest.approx(pooled["B"] / pooled["A"])
    markers = [_numeric_array(collection.get_offsets()) for collection in axis.collections]
    assert len(markers) == 4  # model A then B, chickpea then lentil
    np.testing.assert_allclose([markers[0][0][1], markers[1][0][1]], [raw["A · Chickpea, ID"] / pooled["A"], raw["A · Lentil, ID"] / pooled["A"]])
    for column in range(columns):  # a bar is the mean of its model's two markers
        for model, first in (("A", 0), ("B", 2)):
            bar = bars[(0 if model == "A" else columns) + column]
            assert bar.get_height() == pytest.approx(np.mean([markers[first][column][1], markers[first + 1][column][1]]))
            assert markers[first][column][0] == pytest.approx(bar.get_x() + bar.get_width() / 2)
    plt.close(figure)


def _shift_case_grid(frame: pd.DataFrame, position: int, path: Path) -> None:
    """Move the case at ``position`` onto another grid: its payload, coordinates shifted, is written to ``path`` and bound by digest."""
    with np.load(Path(str(frame.iloc[position]["npz_path"])), allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["coordinates"] = arrays["coordinates"].copy()
    arrays["coordinates"][0] += 0.02
    np.savez_compressed(path, **arrays)
    _record_payload(frame, path)
    frame.loc[frame.index[position], "npz_path"] = str(path)


def test_pooling_refuses_materials_whose_cases_lie_on_different_grids(tmp_path: Path) -> None:
    """Each material passes the grid admission alone and draws separately, but pooled materials must share one grid: both views refuse."""
    frame, _x_grid = steady_frame(tmp_path)
    frame.loc[1, "material_family"] = "chickpea"
    _shift_case_grid(frame, 1, tmp_path / "chickpea_case.npz")
    scope = _scope({"A": frame, "B": _steady_peer(frame)}, comparison=True)
    keys = [item.key for item in scope.inventory.items]
    plots = analysis.evaluation.plots

    separate = scope.frames(selection.resolve_choices(scope.inventory, keys, pooled=False))
    plt.close(plots.sensitivity_capacity.plot_capacity_accuracy(datasets=separate, max_cases=_CASE_COUNT))

    pooled = scope.frames(selection.resolve_choices(scope.inventory, keys, pooled=True))
    for plot in (plots.run_summary.plot_relative_comparison_scoreboard, plots.sensitivity_capacity.plot_capacity_accuracy):
        with pytest.raises(dataframe.ComparisonCompatibilityError, match=r"\['chickpea', 'lentil'\].*mixes incompatible grids"):
            plot(datasets=pooled)


def _set_diagnostics(frame: pd.DataFrame, value: float | None) -> None:
    """Give the task macro and every group of ``frame``'s aggregate the value ``value``, or make the aggregate unavailable for ``None``."""
    aggregate = frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]
    if value is None:
        aggregate.update(
            available=False, value=None, field_metrics=None, relative_unavailable_cases=None, group_statistics=None, finite_value_count=0
        )
        aggregate["nonfinite_value_count"] = aggregate["required_value_count"]
        return
    aggregate["value"] = value
    for statistics in aggregate["group_statistics"].values():
        statistics["normalized_rmse"] = value


@pytest.mark.parametrize(
    ("values", "best", "listed"),
    [
        ({"A": (0.0, 0.0), "B": (1.0, 3.0)}, "zero", "B (Chickpea) 1, B (Lentil) 3"),
        ({"A": (1.0, None), "B": (None, 3.0)}, "unavailable", "A (Chickpea) 1, B (Lentil) 3"),
    ],
    ids=["zero-best", "no-best"],
)
def test_an_undefined_scoreboard_ratio_is_named_with_the_finite_material_values_as_raw_values(
    tmp_path: Path, values: Mapping[str, tuple[float | None, float | None]], best: str, listed: str
) -> None:
    """A best pooled value of zero or none leaves a finite material value without a ratio: the note says why and lists it as a raw value."""
    drawn, _whole_role = _pooled_frames(tmp_path)
    for model, by_material in values.items():
        for material, value in zip(("Chickpea", "Lentil"), by_material, strict=True):
            _set_diagnostics(drawn[f"{model} · {material}, ID"], value)

    figure = analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(datasets=drawn)

    axis = figure.axes[0]
    columns = [label.get_text() for label in axis.get_xticklabels()]
    text = " ".join(_figure_texts(figure))
    for column in columns:
        assert f"Ratio undefined for {column} on ID, all materials (displayed best is {best}); raw values: {listed}" in text
    # A zero best still ranks the exact zeros, model A's, at 1; without any best no bar is drawn.
    assert [cast("Rectangle", patch).get_height() for patch in axis.patches] == ([1.0] * len(columns) if best == "zero" else [])
    plt.close(figure)


def test_capacity_names_a_pooled_point_the_mean_per_material_error_never_the_artifact_objective(tmp_path: Path) -> None:
    """A pooled point is no artifact objective: title and axis name the mean per-material error; separate materials keep their names."""
    drawn, _whole_role = _pooled_frames(tmp_path)
    separate = dataset_cuts.DisplayFrames(drawn, {key: (key.split(" · ")[1], model) for key, (_dataset, model) in drawn.labels.items()})
    plots = analysis.evaluation.plots.sensitivity_capacity

    pooled_figure = plots.plot_capacity_accuracy(datasets=drawn, max_cases=_CASE_COUNT)
    separate_figure = plots.plot_capacity_accuracy(datasets=separate, max_cases=_CASE_COUNT)

    pooled_texts = [*_figure_texts(pooled_figure), pooled_figure.axes[0].get_ylabel()]
    assert pooled_texts[0].startswith("Capacity versus mean per-material normalized group macro RMSE")
    assert pooled_texts[-1].startswith("Mean per-material normalized group macro RMSE")
    assert not [text for text in pooled_texts if re.search("authoritative|aggregate|objective", text, flags=re.IGNORECASE)]
    assert _figure_texts(separate_figure)[0].startswith("Capacity versus authoritative aggregate accuracy")
    assert separate_figure.axes[0].get_ylabel().startswith("Normalized group macro RMSE")
    plt.close(pooled_figure)
    plt.close(separate_figure)


def test_a_capacity_drawn_only_by_a_material_marker_stays_inside_the_axis(tmp_path: Path) -> None:
    """
    Model B (24 parameters) has no chickpea objective, so only its lentil marker is drawn; that capacity sets the axis as a pooled point does.

    Model A's pooled point stands alone at 8, so a range padded around 8 alone would cut B's marker off.
    """
    drawn, _whole_role = _pooled_frames(tmp_path)
    channels = domain.tasks.steady_flow.STEADY_FLOW.output_names
    nonfinite = validity_contract.build_field_prediction_validity(np.full((len(channels), 1, 1), np.nan), channels=channels)
    drawn["B · Chickpea, ID"]["prediction_validity"] = json.dumps(nonfinite)
    for key in ("B · Chickpea, ID", "B · Lentil, ID"):
        drawn[key].attrs["artifact_provenance"]["model"]["parameter_counts"] = {"total": 24, "trainable": 24}

    figure = analysis.evaluation.plots.sensitivity_capacity.plot_capacity_accuracy(datasets=drawn, max_cases=_CASE_COUNT)

    (axis,) = figure.axes
    capacities = [float(_numeric_array(collection.get_offsets())[0, 0]) for collection in axis.collections]
    assert capacities == [8.0, 8.0, 8.0, 24.0]  # A's pooled point and its two materials, then B's lentil marker alone
    low, high = axis.get_xlim()
    assert all(low < capacity < high for capacity in capacities)
    plt.close(figure)


def test_point_keys_show_the_marker_their_models_are_drawn_with(tmp_path: Path) -> None:
    """
    Runs on two materials draw each model's points with its material marker; a model is named «Chickpea» like a material.

    The run behind «Chickpea» is named «C» on lentil, so every point «Chickpea» names carries the chickpea square. In the capacity
    and the Pareto key it keeps that square in its own colour beside the material group's neutral «Chickpea» square, and «B», whose
    points carry two markers, is a neutral circle. The pressure summary keys a model by the marker of its first series: chickpea
    (the material OOD role) is the first dataset, so «Chickpea» and «B» are squares.
    """
    frame, _x_grid = steady_frame(tmp_path)
    peer = _steady_peer(frame)
    frames: dict[str, pd.DataFrame] = {}
    labels: dict[str, tuple[str, str]] = {}
    for dataset, role, material, models in (
        ("chickpea, Material OOD", "material_ood", "chickpea", ("Chickpea", "B")),
        ("lentil, ID", "eval", "lentil", ("C", "B")),
    ):
        for model in models:
            source = peer if model == "B" else frame
            shown = source.copy(deep=True)
            shown.attrs = copy.deepcopy(source.attrs)
            shown.attrs["dataset_role"] = role
            shown.attrs["artifact_provenance"]["split_role"] = role
            shown.attrs["artifact_provenance"]["selection"]["material_families"] = [material]
            frames[f"{model} · {dataset}"], labels[f"{model} · {dataset}"] = shown, (dataset, model)
    materials = dataset_cuts.DisplayFrames(frames, labels)
    styles = analysis.evaluation.plots.series.series_styles(materials)
    squares = [styles[f"{model} · chickpea, Material OOD"].marker for model in ("Chickpea", "B")]
    assert squares == ["s", "s"]
    plots = analysis.evaluation.plots

    capacity = plots.sensitivity_capacity.plot_capacity_accuracy(datasets=materials, max_cases=_CASE_COUNT)
    pareto = plots.run_summary.plot_accuracy_physics_pareto(datasets=materials)
    pressure = plots.physical_consistency.plot_pressure_boundary_summary(datasets=materials)

    square = MarkerStyle("s")
    drawn_square = square.get_path().transformed(square.get_transform())
    np.testing.assert_allclose(_numeric_array(capacity.axes[0].collections[0].get_paths()[0].vertices), _numeric_array(drawn_square.vertices))
    for figure in (capacity, pareto):
        keys = {legend.get_title().get_text(): list(zip(legend.get_texts(), legend.legend_handles, strict=True)) for legend in figure.legends}
        assert {group: [(text.get_text(), cast("Any", handle).get_marker()) for text, handle in entries] for group, entries in keys.items()} == {
            "Model": [("Chickpea", "s"), ("B", "o"), ("C", "o")],
            "Material": [("Chickpea", "s"), ("Lentil", "o")],
        }
        model, material = (cast("Any", keys[group][0][1]) for group in ("Model", "Material"))
        colour = to_rgb(styles["Chickpea · chickpea, Material OOD"].color)
        assert to_rgb(model.get_markeredgecolor()) == colour != to_rgb(material.get_markeredgecolor())
    # B draws chickpea squares and lentil circles, so its key is a neutral circle and a material key tells the shapes apart.
    pressure_keys = {
        legend.get_title().get_text(): [
            (text.get_text(), cast("Any", handle).get_marker()) for text, handle in zip(legend.get_texts(), legend.legend_handles, strict=True)
        ]
        for legend in pressure.legends
    }
    assert pressure_keys == {"Model": [("Chickpea", "s"), ("B", "o"), ("C", "o")], "Material": [("Chickpea", "s"), ("Lentil", "o")]}
    assert [line.get_marker() for axis in pressure.axes[:2] for line in axis.lines] == ["s"] * 4  # the chickpea row
    for figure in (capacity, pareto, pressure):
        plt.close(figure)


def test_a_comparison_draws_every_selected_dataset_with_models_as_series_and_single_model_views_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 17: a comparison view gets one frame per model and dataset keyed «model · dataset»; a single-model view keeps one dataset."""
    frame, _x_grid = steady_frame(tmp_path)
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets=_model_copies(frame, ("A", "B"))),
        panel.EvaluationContext(key="material_ood", label="Material OOD", datasets=_model_copies(frame, ("A", "B"), role="material_ood")),
    ]
    scope = panel._SteadyScope(contexts, comparison=True, refused_roles={})
    drawn: list[Any] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda _view, _plot, **kwargs: drawn.append(kwargs["kwargs"]["datasets"]) or True)

    panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=None, export={})
    panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=None, export={}, single_model=True)

    assert set(drawn[0]) == {f"{model} · {label}" for model in ("A", "B") for label in ("Lentil, ID", "Lentil, Material OOD")}
    for key, (dataset_label, model_label) in drawn[0].labels.items():
        assert key == f"{model_label} · {dataset_label}"
    assert len(drawn[1]) == 1
    assert len({dataset_label for dataset_label, _model in drawn[1].labels.values()}) == 1


def _two_role_comparison_scope(tmp_path: Path, models: tuple[str, ...] = ("A", "B")) -> Any:
    """Return the comparison scope of the same models in an ID and a material-OOD role."""
    frame, _x_grid = steady_frame(tmp_path)
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets=_model_copies(frame, models)),
        panel.EvaluationContext(key="material_ood", label="Material OOD", datasets=_model_copies(frame, models, role="material_ood")),
    ]
    return panel._SteadyScope(contexts, comparison=True, refused_roles={})


def test_a_dataset_and_model_label_pair_that_collides_in_the_comparison_key_is_refused(tmp_path: Path) -> None:
    """Two (model, dataset) pairs that join to one «model · dataset» key raise instead of dropping a frame."""
    scope = _two_role_comparison_scope(tmp_path, ("A", "A · x"))
    choices = selection.resolve_choices(scope.inventory, [item.key for item in scope.inventory.items], pooled=False)
    renamed = [replace(choices[0], label="lentil, ID"), replace(choices[1], label="x · lentil, ID")]

    with pytest.raises(ValueError, match="A · x · lentil, ID"):
        scope._cut(renamed)


def test_a_comparison_draws_one_dataset_and_lists_a_refused_role_as_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With one shared dataset the comparison keys stay the model labels, and a refused role is named as unavailable."""
    frame, _x_grid = steady_frame(tmp_path)
    models = {"A": frame, "B": frame.copy(deep=True)}
    scope = _scope(models, comparison=True, refused={"parameter_ood": "disagreeing membership in B"})
    drawn: list[Any] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda _view, _plot, **kwargs: drawn.append(kwargs["kwargs"]["datasets"]) or True)

    view = panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=None, export={})

    assert [tuple(frames) for frames in drawn] == [("A", "B")]
    assert all(len(cut) == _CASE_COUNT for cut in drawn[0].values())
    assert [model_label for _dataset, model_label in drawn[0].labels.values()] == ["A", "B"]
    texts = [str(getattr(item, "description", None) or getattr(item, "value", "")) for item in _widget_tree(view.widget)]
    assert "<b>Unavailable</b>" in texts
    unavailable = [item for item in _widget_tree(view.widget) if "disagreeing membership in B" in str(getattr(item, "description", ""))]
    assert unavailable
    assert all(item.disabled for item in unavailable)


def test_steady_global_decomposition_capacity_and_map_values(tmp_path: Path) -> None:
    """Protect core global, spatial, architecture, and decomposition values."""
    frame, x_grid = steady_frame(tmp_path)
    datasets = _displayed({"Steady fixture": frame})
    plots = analysis.evaluation.plots
    with analysis.evaluation.session.EvaluationSession(datasets) as session:
        full = session.full_summary(frame).require_spatial()
        prefix = session.prefix_summary(frame, _CASE_COUNT)
    pressure_index = full.grid.fields.index("p")
    absolute_error_mean = full.absolute_error_mean
    signed_error_std = full.signed_error_std
    assert absolute_error_mean is not None
    assert signed_error_std is not None
    np.testing.assert_allclose(absolute_error_mean[pressure_index], 0.75 * x_grid)
    np.testing.assert_allclose(signed_error_std[pressure_index], 0.25 * x_grid)

    overview = plots.run_summary.build_run_summary_table(datasets)
    assert overview.loc["Steady fixture", "task_id"] == "steady_flow"
    assert overview.loc["Steady fixture", "sample_count"] == _CASE_COUNT
    assert overview.loc["Steady fixture", "normalized_group_macro_rmse"] == pytest.approx(frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"])

    global_figure = plots.error_behavior.plot_global_error_metrics(datasets=datasets)
    # One row per norm (L2, then H1) of box plot, density and CDF.
    assert [axis.get_title() for axis in global_figure.axes] == ["Box plot", "Kernel density", "CDF"] * 2
    # The first case's pressure reference (RMS about 0.6 Pa) is below its 1 Pa floor, so only the second case has relative values.
    relative_l2, relative_h1 = (frame[column].dropna().to_numpy() for column in ("physical_relative_l2", "physical_relative_h1"))
    assert (relative_l2.size, relative_h1.size) == (1, 1)
    cdf_values, cdf_levels = _ecdf_steps(global_figure.axes[2].lines[0])
    np.testing.assert_allclose(cdf_values, relative_l2)
    np.testing.assert_allclose(cdf_levels, (1.0,))
    np.testing.assert_allclose(_ecdf_steps(global_figure.axes[5].lines[0])[0], relative_h1)
    plt.close(global_figure)

    distribution = plots.error_behavior.plot_predictive_error_distributions(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    cdf_values, cdf_levels = _ecdf_steps(distribution.axes[0].lines[0])
    np.testing.assert_allclose(cdf_values, relative_l2)
    np.testing.assert_allclose(cdf_levels, (1.0,))
    for axis, field in zip(  # one row per field below the casewise row
        distribution.axes[1:],
        analysis.evaluation.presentation.display_fields(frame),
        strict=True,
    ):
        source = prefix.magnitudes["velocity"].local_relative_error if field.is_magnitude else prefix.local_relative_error[field.key]
        np.testing.assert_allclose(_numeric_array(axis.lines[0].get_xdata()), source.quantiles)
        np.testing.assert_allclose(_numeric_array(axis.lines[0].get_ydata()), source.probabilities)
    plt.close(distribution)

    bias = plots.error_behavior.plot_mean_field_bias(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(bias.axes[0].collections[0].get_offsets()),
        np.asarray(((0.5, 1.0), (1.0, 2.0))),
    )
    plt.close(bias)

    for figure, title in (
        (
            plots.error_behavior.plot_mean_error_maps(
                datasets=datasets,
                max_cases=_CASE_COUNT,
                error_mode="MAE",
            ),
            "Mean absolute error maps",
        ),
        (
            plots.error_behavior.plot_std_error_maps(
                datasets=datasets,
                max_cases=_CASE_COUNT,
            ),
            "Error standard-deviation maps",
        ),
    ):
        assert title in figure.get_suptitle()
        assert len([text for axis in figure.axes for text in axis.texts if text.get_gid() == "channel-row-label"]) == 4  # a row per field
        plt.close(figure)

    architecture = plots.run_summary.build_architecture_table(datasets)
    assert architecture.loc["Steady fixture", "Architecture · Model family"] == "FNO"
    assert architecture.loc["Steady fixture", "Runtime/evidence · Trainable parameters"] == 8
    capacity = plots.sensitivity_capacity.plot_capacity_accuracy(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(capacity.axes[0].collections[0].get_offsets()),
        np.asarray(((8.0, frame.attrs[dataframe.PRIMARY_OBJECTIVE_ID]["value"]),), dtype=float),
    )
    plt.close(capacity)

    magnitude = plots.error_behavior.plot_error_vs_target_magnitude(
        datasets=datasets,
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(magnitude.axes[0].lines[0].get_xdata()),
        prefix.target_magnitude_error["p"].centers,
    )
    np.testing.assert_allclose(
        _numeric_array(magnitude.axes[0].lines[0].get_ydata()),
        prefix.target_magnitude_error["p"].medians,
    )
    plt.close(magnitude)

    boundary = plots.error_behavior.plot_boundary_error_decomposition(
        datasets=datasets,
        max_cases=_CASE_COUNT,
        channels=("p",),
    )
    np.testing.assert_allclose(
        tuple(cast("Rectangle", patch).get_height() for patch in boundary.axes[0].patches),
        (1.25, np.nan, np.nan, 2.0, 1.0),
        equal_nan=True,
    )
    plt.close(boundary)


def _two_models_on_two_datasets(tmp_path: Path) -> dataset_cuts.DisplayFrames:
    """Return models A and B on the datasets ID and OOD (a material-OOD role), every frame keyed ``<model> · <dataset>``."""
    frame, _x_grid = steady_frame(tmp_path)
    peer = _steady_peer(frame)
    roles = {"ID": "eval", "OOD": "material_ood"}
    cells = []
    for dataset, role in roles.items():
        for model, source in (("A", frame), ("B", peer)):
            cut = source.copy(deep=True)
            cut.attrs = copy.deepcopy(source.attrs)
            cut.attrs["dataset_role"] = role
            cut.attrs["artifact_provenance"]["split_role"] = role
            cells.append((model, dataset, cut))
    return dataset_cuts.DisplayFrames(
        {f"{model} · {dataset}": source for model, dataset, source in cells},
        {f"{model} · {dataset}": (dataset, model) for model, dataset, _source in cells},
    )


def test_steady_line_bar_and_distribution_views_draw_datasets_as_panels_and_models_as_series(tmp_path: Path) -> None:
    """Rule 17: a panel per dataset, the models inside it in their fixed style, one legend for the models and a header per dataset row."""
    shown = _two_models_on_two_datasets(tmp_path)
    styles = analysis.evaluation.plots.series.series_styles(shown)
    # Two runs of one architecture differ by shade; a run keeps its colour on every dataset, and the role is the line style.
    assert styles["A · ID"].color != styles["B · ID"].color
    assert styles["A · OOD"].color == styles["A · ID"].color
    assert styles["A · OOD"].linestyle != styles["A · ID"].linestyle
    plots = analysis.evaluation.plots

    magnitude = plots.error_behavior.plot_error_vs_target_magnitude(datasets=shown, max_cases=_CASE_COUNT)
    columns = [axis for axis in magnitude.axes if axis.get_title()]
    assert [axis.get_title() for axis in columns] == ["ID", "OOD"]  # the dataset labels head the top row only
    for column, dataset in enumerate(("ID", "OOD")):
        for axis in magnitude.axes[column::2]:
            assert [(line.get_color(), line.get_linestyle()) for line in axis.lines] == [
                (styles[f"{model} · {dataset}"].color, styles[f"{model} · {dataset}"].linestyle) for model in ("A", "B")
            ]
    # One entry per model in its run colour, one per role in its line style: no panel's style stands for the others.
    legends = {legend.get_title().get_text(): legend for legend in magnitude.legends}
    assert [text.get_text() for text in legends["Model"].get_texts()] == ["A", "B"]
    assert [cast("Any", handle).get_color() for handle in legends["Model"].legend_handles] == [styles["A · ID"].color, styles["B · ID"].color]
    assert [text.get_text() for text in legends["Role"].get_texts()] == ["ID", "Material OOD"]
    assert [cast("Any", handle).get_linestyle() for handle in legends["Role"].legend_handles] == ["-", ":"]
    plt.close(magnitude)

    metrics = plots.error_behavior.plot_global_error_metrics(datasets=shown)
    assert len(metrics.axes) == 3 * 2 * 2  # box, density and CDF per norm and dataset
    headers = [text for text in _figure_texts(metrics) if text.endswith(("Relative L2", "Relative H1"))]
    assert headers == ["ID · Relative L2", "ID · Relative H1", "OOD · Relative L2", "OOD · Relative H1"]
    plt.close(metrics)

    scoreboard = plots.run_summary.plot_relative_comparison_scoreboard(datasets=shown)
    panels = scoreboard.axes
    assert [len(axis.patches) for axis in panels] == [2 * len(panels[0].get_xticklabels())] * 2  # two models' bars per diagnostic
    first = len(panels[0].get_xticklabels())  # model A's bars come first, then model B's
    assert [(axis.patches[0].get_facecolor()[:3], axis.patches[first].get_facecolor()[:3]) for axis in panels] == [
        (to_rgb(styles["A · ID"].color), to_rgb(styles["B · ID"].color))
    ] * 2
    plt.close(scoreboard)

    heatmap = plots.sensitivity_capacity.plot_metadata_error_heatmap(datasets=shown, max_cases=_CASE_COUNT)
    assert [len(axis.images) for axis in heatmap.axes[:4]] == [1] * 4  # a column per dataset, a row per run
    assert [axis.get_title() for axis in heatmap.axes[:4]] == ["ID", "OOD", "", ""]  # the datasets head the top row
    # Each run heads its own row, on the row's first panel.
    assert [[text.get_text() for text in axis.texts if text.get_text() in {"A", "B"}] for axis in heatmap.axes[:4]] == [["A"], [], ["B"], []]
    heatmap.canvas.draw()
    panels = [axis.get_position() for axis in heatmap.axes[:4]]
    bar = heatmap.axes[-1].get_position()
    assert (bar.y0, bar.y1) == pytest.approx((min(box.y0 for box in panels), max(box.y1 for box in panels)))  # one bar spans the block
    plt.close(heatmap)


def test_steady_views_draw_an_ood_dataset_of_one_model_and_name_a_single_dataset(tmp_path: Path) -> None:
    """An OOD role needs no marker of its own, and the one dataset of a figure is still named in its row headers."""
    frame, _x_grid = steady_frame(tmp_path)
    frame.attrs["dataset_role"] = "parameter_ood"
    shown = _displayed({"chickpea, OOD": frame})
    plots = analysis.evaluation.plots
    plt.close(plots.sensitivity_capacity.plot_capacity_accuracy(datasets=shown, max_cases=_CASE_COUNT))
    plt.close(plots.run_summary.plot_accuracy_physics_pareto(datasets=shown))
    metrics = plots.error_behavior.plot_global_error_metrics(datasets=shown)
    headers = [text for text in _figure_texts(metrics) if text.endswith(("Relative L2", "Relative H1"))]
    assert headers == ["chickpea, OOD · Relative L2", "chickpea, OOD · Relative H1"]
    plt.close(metrics)


def test_steady_single_panel_legends_stay_inside_and_trends_give_each_model_a_row(tmp_path: Path) -> None:
    """One data panel keeps its legend inside, with the title of a single group; several models are rows of the trends."""
    frame, _x_grid = steady_frame(tmp_path)
    two = _displayed({"A": frame, "B": _steady_peer(frame)}, dataset="ID")
    plots = analysis.evaluation.plots
    capacity = plots.sensitivity_capacity.plot_capacity_accuracy(datasets=two, max_cases=_CASE_COUNT)
    assert len(capacity.axes) == 1  # one dataset is one panel; the configuration is its own table view
    assert capacity.axes[0].get_legend() is not None
    assert not capacity.legends
    plt.close(capacity)
    boundary = plots.error_behavior.plot_boundary_error_decomposition(datasets=_displayed({"A": frame}), max_cases=_CASE_COUNT)
    assert cast("Any", boundary.axes[0].get_legend()).get_title().get_text() == "x-distance from left/right boundary"
    plt.close(boundary)
    trends = plots.sensitivity_capacity.plot_metadata_error_trends(datasets=two, max_cases=_CASE_COUNT, channels=("p",))
    assert [axis.get_title() for axis in trends.axes] == ["ID", ""]  # a row per model, the dataset heads the top row
    assert all(line.get_color() == trends.axes[0].lines[0].get_color() for axis in trends.axes for line in axis.lines)  # colour is the channel
    plt.close(trends)


def test_the_role_hatch_contrasts_with_its_face_on_every_boundary_band(tmp_path: Path) -> None:
    """The top band is the model colour itself, so its hatch must not be drawn in that colour."""
    frame, _x_grid = steady_frame(tmp_path)
    frame.attrs["dataset_role"] = "parameter_ood"
    figure = analysis.evaluation.plots.error_behavior.plot_boundary_error_decomposition(
        datasets=_displayed({"A": frame}), max_cases=_CASE_COUNT, channels=("p",)
    )
    bars = figure.axes[0].patches  # one channel: one bar per band, the top band last
    assert all(bar.get_hatch() for bar in bars)
    assert all(bar.get_edgecolor() != bar.get_facecolor() for bar in bars)
    plt.close(figure)


def _runs_of_one_architecture(frame: pd.DataFrame, count: int) -> dataset_cuts.DisplayFrames:
    """Return ``count`` runs of one architecture as the models of the dataset ID."""
    runs = {}
    for index in range(count):
        run = _steady_peer(frame)
        provenance = run.attrs["artifact_provenance"]["run"]
        provenance.update(
            {"name": f"run-{index}", "effective_config_digest": f"{index:02x}" * 32, "best_checkpoint_sha256": f"{index + 64:02x}" * 32}
        )
        runs[f"R{index}"] = run
    return _displayed(runs, dataset="ID")


def test_trends_colour_the_channel_so_many_runs_never_hit_the_run_colour_limit(tmp_path: Path) -> None:
    """Eleven runs of one architecture exceed the run palette, yet the trends, whose colour is the channel, draw."""
    frame, _x_grid = steady_frame(tmp_path)
    figure = analysis.evaluation.plots.sensitivity_capacity.plot_metadata_error_trends(
        datasets=_runs_of_one_architecture(frame, 11), max_cases=_CASE_COUNT, channels=("p",)
    )
    assert len(figure.axes) == 11  # a row per run
    plt.close(figure)


def test_four_runs_of_one_architecture_take_run_colours_and_the_figure_names_it(tmp_path: Path) -> None:
    """Beyond the shade limit every run gets its own palette colour, and the figure note says that colour means the run."""
    frame, _x_grid = steady_frame(tmp_path)
    runs = _runs_of_one_architecture(frame, 4)
    shown = runs
    styles = analysis.evaluation.plots.series.series_styles(shown)
    assert styles.run_colours
    assert len({style.color for style in styles.values()}) == len(runs)
    figure = analysis.evaluation.plots.error_behavior.plot_error_vs_target_magnitude(datasets=shown, max_cases=_CASE_COUNT)
    assert any("colour = run" in text for text in _figure_texts(figure))
    plt.close(figure)


def test_steady_error_titles_follow_the_label_style(tmp_path: Path) -> None:
    """The metric labels of the line views come from the active label style, in every style."""
    shown = _displayed({"A": steady_frame(tmp_path)[0]})
    fields = analysis.evaluation.presentation.display_fields(shown["A"])
    try:
        for style in ("symbol", "name", "name_symbol"):
            analysis.presentation.field_labels.set_label_style(style)
            magnitude = analysis.evaluation.plots.error_behavior.plot_error_vs_target_magnitude(datasets=shown, max_cases=_CASE_COUNT)
            assert [axis.get_ylabel() for axis in magnitude.axes] == [f"Median {field.metric_label('MAE')} [{field.unit}]" for field in fields]
            plt.close(magnitude)
    finally:
        analysis.presentation.field_labels.set_label_style("symbol")


def test_steady_relative_errors_below_the_field_floor_are_unavailable_and_counted(tmp_path: Path) -> None:
    """
    A case whose reference RMS lies below the field's floor has no relative error in maps or distributions; it is counted.

    The first fixture case has a pressure reference RMS of about 0.6 Pa, below the 1 Pa floor; the second case
    (about 1.2 Pa) and every velocity reference lie above their floors.
    """
    frame, x_grid = steady_frame(tmp_path)
    with analysis.evaluation.session.EvaluationSession({"fixture": frame}) as session:
        full = session.full_summary(frame).require_spatial()
        prefix = session.prefix_summary(frame, _CASE_COUNT)
    pressure = prefix.local_relative_error["p"]
    assert pressure.source_point_count == x_grid.size
    assert pressure.unavailable_case_count == 1
    assert prefix.local_relative_error["u"].unavailable_case_count == 0
    assert prefix.magnitudes["velocity"].local_relative_error.unavailable_case_count == 0
    unavailable_cases, relative_mean = full.local_relative_unavailable_cases, full.local_relative_error_mean
    assert unavailable_cases is not None
    assert relative_mean is not None
    assert dict(zip(full.grid.fields, unavailable_cases.tolist(), strict=True)) == {"p": 1, "u": 0, "v": 0}
    assert full.magnitudes["velocity"].local_relative_unavailable_cases == 0
    np.testing.assert_allclose(relative_mean[full.grid.fields.index("p")], x_grid / np.sqrt(np.mean(x_grid**2)))

    overview = analysis.evaluation.plots.samples_outliers.plot_sample_prediction_overview(
        datasets={"fixture": frame}, row_position=0, error_mode="Relative [%]"
    )
    unavailable = [text.get_text() for axis in overview.axes for text in axis.texts if text.get_text().startswith("Unavailable")]
    assert len(unavailable) == 1
    assert "floor" in unavailable[0]
    plt.close(overview)


def test_relative_norm_distributions_name_their_available_and_unavailable_cases(tmp_path: Path) -> None:
    """The per-case relative-norm distributions state how many cases they show and how many are unavailable."""
    frame, _x_grid = steady_frame(tmp_path)
    datasets = _displayed({"Steady fixture": frame})
    plots = analysis.evaluation.plots.error_behavior
    # The first case's pressure reference lies below its floor, so one of the two cases has a relative norm.
    global_figure = plots.plot_global_error_metrics(datasets=datasets)
    for norm in ("Relative L2", "Relative H1"):
        assert any(f"{norm}: 1 of {_CASE_COUNT} cases, 1 unavailable" in text for text in _figure_texts(global_figure)), norm
    plt.close(global_figure)
    distribution = plots.plot_predictive_error_distributions(datasets=datasets, max_cases=_CASE_COUNT)
    assert any(f"Relative L2: 1 of {_CASE_COUNT} cases, 1 unavailable" in text.get_text() for text in distribution.axes[0].texts)
    assert distribution.axes[0].get_xscale() == "log"
    plt.close(distribution)
    # An exact prediction has a relative L2 of 0, which a log axis could not draw: that column starts at 0 on symlog.
    exact = frame.copy()
    exact.attrs = frame.attrs
    exact["physical_relative_l2"] = exact["physical_relative_l2"].where(exact["physical_relative_l2"].isna(), 0.0)
    distribution = plots.plot_predictive_error_distributions(datasets=_displayed({"Steady fixture": exact}), max_cases=_CASE_COUNT)
    assert (distribution.axes[0].get_xscale(), distribution.axes[0].get_xlim()[0]) == ("symlog", 0.0)
    # The single available case is a visible rise of the CDF from 0 to 1 at its value, not a lone invisible point.
    cdf = _numeric_array(distribution.axes[0].lines[0].get_ydata())
    assert (cdf.min(), cdf.max()) == (0.0, 1.0)
    plt.close(distribution)


def test_pressure_drop_mismatch_divides_by_the_declared_drop_and_refuses_a_non_positive_one(tmp_path: Path) -> None:
    """The relative pressure-drop mismatch divides by the positive declared drop; a non-positive drop fails closed."""
    for name in ("positive", "flat"):
        (tmp_path / name).mkdir()
    frame, _x_grid = steady_frame(tmp_path / "positive")
    pressure = analysis.evaluation.plots.physical_consistency.plot_pressure_consistency(
        datasets=_displayed({"fixture": frame}), max_cases=_CASE_COUNT
    )
    # Declared drop 1 Pa, predicted inlet and outlet rows share one mean: a mismatch of 1 Pa in both cases. The tolerance
    # is below 1e-12, so a denominator stabilised with an epsilon would fail.
    np.testing.assert_allclose(_ecdf_steps(pressure.axes[0].lines[0])[0], (1.0, 1.0), rtol=1e-14, atol=0.0)
    plt.close(pressure)
    flat, _x_grid = steady_frame(tmp_path / "flat", pressure_drop=0.0)
    with pytest.raises(ValueError, match="declared pressure drop"):
        analysis.evaluation.plots.physical_consistency.plot_pressure_consistency(datasets=_displayed({"fixture": flat}), max_cases=_CASE_COUNT)


def test_steady_physical_spectral_and_sensitivity_values(tmp_path: Path) -> None:
    """Protect steady-only physics, spectral transfer, and parameter evidence."""
    frame, _x_grid = steady_frame(tmp_path)
    datasets = {"Steady fixture": frame}
    plots = analysis.evaluation.plots

    physical = plots.physical_consistency.build_physical_consistency_summary_table(datasets)
    expected_medians = {
        "momentum_residual_mse median": 12.5,
        "flux_balance_mse median": 0.025,
        "pressure_boundary_mse median": 2.5,
    }
    for column, expected in expected_medians.items():
        assert physical.loc["Steady fixture", column] == pytest.approx(expected)

    momentum = plots.physical_consistency.plot_brinkman_residual(datasets=datasets, max_cases=_CASE_COUNT)
    cdf_axis = momentum.axes[-1]  # the band above the maps, added after the map grid
    cdf_values, cdf_levels = _ecdf_steps(cdf_axis.lines[0])
    np.testing.assert_allclose(cdf_values, (5.0, 20.0))
    np.testing.assert_allclose(cdf_levels, (0.5, 1.0))
    np.testing.assert_allclose(_numeric_array(momentum.axes[0].collections[0].get_array()), 1.5 * np.sqrt(5.0))
    plt.close(momentum)

    flux = plots.physical_consistency.plot_flux_balance(datasets=_displayed(datasets), max_cases=_CASE_COUNT)
    np.testing.assert_allclose(_ecdf_steps(flux.axes[0].lines[0])[0], (0.01, 0.04))
    np.testing.assert_allclose(_numeric_array(flux.axes[1].lines[0].get_xdata()), 0.15)
    plt.close(flux)

    cdf_grid = plots.physical_consistency.plot_physical_consistency_cdf_grid(
        datasets=_displayed(datasets),
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_ecdf_steps(cdf_grid.axes[0].lines[0])[0], (0.01, 0.04))
    np.testing.assert_allclose(_ecdf_steps(cdf_grid.axes[1].lines[0])[0], (5.0, 20.0))
    np.testing.assert_allclose(_ecdf_steps(cdf_grid.axes[3].lines[0])[0], (1.0, 4.0))
    plt.close(cdf_grid)

    pressure = plots.physical_consistency.plot_pressure_consistency(
        datasets=_displayed(datasets),
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_ecdf_steps(pressure.axes[1].lines[0])[0], (1.0, 4.0))
    assert pressure.axes[1].get_xlabel().endswith("[-]")
    plt.close(pressure)

    demand = plots.spectral_fidelity.plot_spectral_demand_prediction_error(datasets=_displayed(datasets), max_cases=_CASE_COUNT)
    reference_power = _numeric_array(_medians(demand.axes[0])[0].get_ydata())
    prediction_power = _numeric_array(_medians(demand.axes[0])[1].get_ydata())
    error_power = _numeric_array(_medians(demand.axes[1])[0].get_ydata())
    # The fixture's prediction is 4 x the reference and its error equals the reference, so the unit-sum spectra coincide.
    np.testing.assert_allclose(prediction_power, reference_power)
    np.testing.assert_allclose(error_power, reference_power)
    assert demand.axes[0].get_ylabel() == "Normalized spectral power [-]"
    plt.close(demand)

    transfer = plots.spectral_fidelity.plot_spectral_transfer_ratio(
        datasets=_displayed(datasets),
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(
        _numeric_array(transfer.axes[0].lines[0].get_ydata()),
        np.log10(4.0),
    )
    np.testing.assert_allclose(_numeric_array(transfer.axes[1].lines[0].get_ydata()), 1.0)
    plt.close(transfer)

    heatmap = plots.sensitivity_capacity.plot_metadata_error_heatmap(
        datasets=_displayed(datasets),
        max_cases=_CASE_COUNT,
    )
    np.testing.assert_allclose(_numeric_array(heatmap.axes[0].images[0].get_array()), 1.0)
    assert heatmap.axes[-1].get_ylabel() == "Spearman correlation [-]"
    plt.close(heatmap)

    trend = plots.sensitivity_capacity.plot_metadata_error_trends(
        datasets=_displayed(datasets),
        max_cases=_CASE_COUNT,
        channels=("p",),
    )
    assert np.isnan(_numeric_array(trend.axes[0].lines[0].get_xdata())).all()
    assert any(text.get_text() == "No resolved sensitivity for these cases" for text in trend.axes[0].texts)
    plt.close(trend)


def _nonfinite_frame(tmp_path: Path, positions: Sequence[int]) -> pd.DataFrame:
    """Return the two-case steady fixture with the given rows rewritten as NONFINITE predictions."""
    tmp_path.mkdir()
    frame, _grid = steady_frame(tmp_path)
    channels = domain.tasks.steady_flow.STEADY_FLOW.output_names
    for position in positions:
        path = Path(str(frame.loc[position, "npz_path"]))
        with np.load(path) as payload:
            arrays = dict(payload)
        arrays["pred"][0, 2, 3] = np.nan
        arrays["err"][0, 2, 3] = np.nan
        np.savez_compressed(path, **arrays)
        _record_payload(frame, path)
        validity = validity_contract.build_field_prediction_validity(arrays["pred"][: len(channels)], channels=channels)
        frame.loc[position, "prediction_validity"] = json.dumps(validity)
        frame.loc[position, ["physical_relative_l2", "physical_relative_h1"]] = np.nan
    frame.attrs["artifact_provenance"]["evaluation"]["prediction_validity"] = validity_contract.table_role_prediction_validity(frame)
    return frame


def test_prefix_summary_leaves_nonfinite_cases_out_whole_and_counts_them(tmp_path: Path) -> None:
    """A NONFINITE case never reaches a reduction: the rest reduces as if it were alone, and a prefix without finite case is explicit."""
    finite = _nonfinite_frame(tmp_path / "finite", ())
    mixed = _nonfinite_frame(tmp_path / "mixed", (1,))
    none = _nonfinite_frame(tmp_path / "none", (0, 1))
    with analysis.evaluation.session.EvaluationSession({"finite": finite}) as session:
        alone = session.prefix_summary(finite, 1)
    with analysis.evaluation.session.EvaluationSession({"mixed": mixed}) as session:
        partial = session.prefix_summary(mixed, _CASE_COUNT)
    with analysis.evaluation.session.EvaluationSession({"none": none}) as session:
        empty = session.prefix_summary(none, _CASE_COUNT)

    assert (partial.case_count, partial.left_out_count, partial.used_count) == (_CASE_COUNT, 1, 1)
    for field in partial.fields:
        np.testing.assert_array_equal(partial.local_relative_error[field].quantiles, alone.local_relative_error[field].quantiles)
        np.testing.assert_array_equal(partial.target_magnitude_error[field].medians, alone.target_magnitude_error[field].medians)
        np.testing.assert_array_equal(partial.spectra[field].error, alone.spectra[field].error)
    assert partial.spectra["p"].error.shape[0] == 1
    assert (empty.case_count, empty.left_out_count, empty.used_count) == (_CASE_COUNT, _CASE_COUNT, 0)
    assert empty.fields == alone.fields
    reductions = (empty.local_relative_error, empty.target_magnitude_error, empty.boundary_distance_error, empty.spectra, empty.magnitudes)
    assert not any(reductions)


def test_full_summary_leaves_a_nonfinite_case_out_of_the_local_relative_maps_whole(tmp_path: Path) -> None:
    """A NONFINITE case neither voids a local relative map nor counts as below a floor: the maps are those of the finite case alone."""
    finite = _nonfinite_frame(tmp_path / "finite", ())
    mixed = _nonfinite_frame(tmp_path / "mixed", (1,))  # the second case is the only one above the pressure floor
    with analysis.evaluation.session.EvaluationSession({"finite": finite}) as session:
        alone = session.full_summary(finite, 1).require_spatial()
    with analysis.evaluation.session.EvaluationSession({"mixed": mixed}) as session:
        partial = session.full_summary(mixed).require_spatial()

    assert partial.sample_count == _CASE_COUNT
    assert (partial.local_relative_nonfinite_cases, alone.local_relative_nonfinite_cases) == (1, 0)  # a plot can name the left-out case
    for name in ("local_relative_error_mean", "local_relative_unavailable_cases"):
        np.testing.assert_array_equal(np.asarray(getattr(partial, name)), np.asarray(getattr(alone, name)))
    for group, magnitude in partial.magnitudes.items():
        np.testing.assert_array_equal(magnitude.local_relative_error_mean, alone.magnitudes[group].local_relative_error_mean)
        assert magnitude.local_relative_unavailable_cases == alone.magnitudes[group].local_relative_unavailable_cases


def test_the_local_relative_maps_name_the_nonfinite_case_they_leave_out(tmp_path: Path) -> None:
    """The relative mean maps say that the NONFINITE case is left out; the signed maps, which keep its finite cells, say nothing."""
    mixed = _nonfinite_frame(tmp_path / "mixed", (1,))
    behavior = analysis.evaluation.plots.error_behavior
    notes = {}
    for mode in (error_diagnostics.SIGNED_ERROR, "Relative [%]"):
        figure = behavior.plot_mean_error_maps(datasets={"mixed": mixed}, max_cases=_CASE_COUNT, error_mode=mode)
        notes[mode] = [text.get_text() for text in figure.findobj(Text) if "non-finite, left out" in text.get_text()]
        plt.close(figure)
    assert notes[error_diagnostics.SIGNED_ERROR] == []
    assert notes["Relative [%]"]
    assert all(f"1 of {_CASE_COUNT} cases non-finite, left out" in note for note in notes["Relative [%]"])


def test_prefix_plots_show_an_empty_prefix_instead_of_failing(tmp_path: Path) -> None:
    """Every prefix consumer states `unavailable` when no prefix case is finite, and the curated spectral entry still renders the rest."""
    plots = analysis.evaluation.plots
    none = _displayed({"No finite case": _nonfinite_frame(tmp_path / "none", (0, 1))})
    mixed = _displayed({"One finite case": _nonfinite_frame(tmp_path / "mixed", (1,))})
    renderers = (
        plots.spectral_fidelity.plot_spectral_demand_prediction_error,
        plots.spectral_fidelity.plot_spectral_transfer_ratio,
        plots.error_behavior.plot_predictive_error_distributions,
        plots.error_behavior.plot_error_vs_target_magnitude,
        plots.error_behavior.plot_boundary_error_decomposition,
    )
    for render in renderers:
        figure = render(datasets=none)
        assert any(f"unavailable (0 finite of {_CASE_COUNT} prefix cases)" in text for text in _figure_texts(figure)), render.__name__
        plt.close(figure)
    figure = plots.spectral_fidelity.plot_spectral_demand_prediction_error(datasets=mixed)
    assert figure.axes[0].lines
    plt.close(figure)
    for render in (plots.error_behavior.plot_predictive_error_distributions, plots.error_behavior.plot_global_error_metrics):
        plt.close(render(datasets=mixed))
    distribution = plots.error_behavior.plot_predictive_error_distributions(datasets=none)
    assert all(any("unavailable (0 finite" in text.get_text() for text in axis.texts) for axis in distribution.axes)  # on every panel
    plt.close(distribution)
    figure = plots.error_behavior.plot_global_error_metrics(datasets=none)
    assert any(f"Relative L2: 0 of {_CASE_COUNT} cases, {_CASE_COUNT} unavailable" in text for text in _figure_texts(figure))
    plt.close(figure)


def test_a_compared_model_without_a_finite_case_stays_in_the_title_and_its_panels_say_so(tmp_path: Path) -> None:
    """Model B has no finite prefix case: the distributions still end with «(A vs B)», and B's panels name it as unavailable."""
    frames = {"A": _nonfinite_frame(tmp_path / "a", ()), "B": _steady_peer(_nonfinite_frame(tmp_path / "b", (0, 1)))}
    figure = analysis.evaluation.plots.error_behavior.plot_predictive_error_distributions(datasets=_displayed(frames, dataset="ID"))
    texts = _figure_texts(figure)
    assert texts[0].endswith(" (A vs B)")
    assert any(f"B: unavailable (0 finite of {_CASE_COUNT} prefix cases)" in text for text in texts)
    plt.close(figure)


@pytest.mark.parametrize(
    ("models", "cut", "save"),
    [
        (_MANY_MODELS[:8], False, False),
        (("x" * 300,), False, False),
        (_MANY_MODELS, True, False),
        (_MANY_MODELS, True, True),
    ],
    ids=["wrapped", "unbroken-word", "cut-at-smallest-font", "saved-tight"],
)
def test_the_empty_prefix_lines_render_inside_their_panel(models: tuple[str, ...], cut: bool, save: bool) -> None:
    """
    Models without a finite prefix case: drawn, or saved with a tight box, the text lies inside the panel.

    Every line is shown, broken at words and a word wider than the panel at characters, unless even the smallest font cannot hold
    the rows: the rows that fit then end in one row «… k more».
    """
    layout = analysis.evaluation.plots.layout
    figure, axis = plt.subplots(figsize=(4.0, 3.0))
    for model in models:
        layout.mark_empty_prefix(axis, case_count=_CASE_COUNT, model=model)
    if save:
        figure.savefig(io.BytesIO(), bbox_inches="tight")
    else:
        figure.canvas.draw()
    panel = axis.get_window_extent()
    (text,) = axis.texts
    extent = text.get_window_extent()
    assert (panel.x0 <= extent.x0, extent.x1 <= panel.x1, panel.y0 <= extent.y0, extent.y1 <= panel.y1) == (True,) * 4
    *shown, last = text.get_text().splitlines()
    if cut:
        count = re.fullmatch("\N{HORIZONTAL ELLIPSIS} ([0-9]+) more", last)
        assert count is not None
        assert shown[0].startswith("model 0:")
        assert len(shown) + int(count.group(1)) >= len(models)
    else:
        assert "".join(text.get_text().split()) == "".join(f"{model}:{layout.empty_prefix_text(_CASE_COUNT)}".replace(" ", "") for model in models)
    plt.close(figure)


def test_steady_sample_kappa_outlier_and_comparison_values(tmp_path: Path) -> None:
    """Protect exact selected cases, Kappa evidence, rankings, and paired fields."""
    frame, x_grid = steady_frame(tmp_path)
    peer = _steady_peer(frame)
    datasets = {"lentil, ID": frame}  # keyed by its dataset label as in a single-model panel
    plots = analysis.evaluation.plots.samples_outliers
    case = cases.load_case(frame, 0)
    assert case.permeability is not None
    np.testing.assert_allclose(case.permeability[0], 1.0e-4)
    np.testing.assert_allclose(case.permeability[1], 0.0)
    np.testing.assert_allclose(case.permeability[2], 2.0e-4)
    np.testing.assert_allclose(case.reference[0], 0.5 * x_grid)
    np.testing.assert_allclose(case.prediction[0], x_grid)

    linked_outlier = plots.plot_linked_outlier_cases(
        datasets=datasets,
        selection_index=0,
        channel="p",
    )
    assert linked_outlier.get_suptitle() == "$p$ outlier field view · Lentil, ID — Case 2"
    plt.close(linked_outlier)

    linked_extreme = plots.plot_linked_input_extreme_cases(
        datasets=datasets,
        selection_index=0,
        parameter="parameters_bed.structure.coarse_len_rel",
    )
    assert linked_extreme.get_suptitle() == "Input extreme: Coarse correlation length minimum · Lentil, ID — Case 1"
    plt.close(linked_extreme)

    models = {"Model A": frame, "Model B": peer}
    comparison = plots.plot_pressure_velocity_comparison(datasets=models, row_position=0)
    assert comparison.get_suptitle() == "Pressure and velocity comparison · Lentil, ID — Case 1"
    # Rows p and |u|, columns the reference then every model, each header once above its column and every map with its own bar.
    maps, bars = (np.asarray(axes, dtype=object).reshape(2, 3) for axes in (comparison.axes[:6], comparison.axes[6:]))
    assert [axis.get_title() for axis in maps[0]] == ["Reference", "Model A", "Model B"]
    assert not any(axis.get_title() for axis in maps[1])
    assert [bar.get_ylabel() for bar in bars.flat] == ["[Pa]"] * 3 + ["[m/s]"] * 3
    labels = [text.get_text() for axis in maps[:, 0] for text in axis.texts if text.get_gid() == "channel-row-label"]
    assert labels[0] == r"$\mathbf{p}$"
    assert r"|\mathbf{u}|" in labels[1]
    plt.close(comparison)
    chosen = plots.plot_pressure_velocity_comparison(datasets=models, row_position=0, models=("Model B",), scale_mode="Shared (GT)")
    assert [axis.get_title() for axis in chosen.axes[:4] if axis.get_title()] == ["Reference", "Model B"]
    assert len(chosen.axes) == 8
    plt.close(chosen)
    with pytest.raises(ValueError, match="distinct displayed models"):
        plots.plot_pressure_velocity_comparison(datasets=models, models=("Model A", "Model A"))

    scoreboard = analysis.evaluation.plots.run_summary.plot_relative_comparison_scoreboard(
        datasets=_displayed(models, dataset="ID"),
    )
    assert any("Comparison-relative normalized error" in text for text in _figure_texts(scoreboard))
    plt.close(scoreboard)
    pareto = analysis.evaluation.plots.run_summary.plot_accuracy_physics_pareto(
        datasets=_displayed(models, dataset="ID"),
    )
    assert tuple(axis.get_title() for axis in pareto.axes[:3]) == ("Momentum residual", "Flux balance", "Pressure boundary")
    plt.close(pareto)


def test_steady_timing_binding_rejects_identity_drift(tmp_path: Path) -> None:
    """The required timing report must name the artifact's exact Dataset and checkpoint."""
    frame, _x_grid = steady_frame(tmp_path)
    provenance = frame.attrs["artifact_provenance"]
    payload = provenance["timing"]["report"]
    dataset = provenance["dataset"]
    role = _saved_role("eval", dataset["name"], dict(zip(dataset["source_dataset_ids"], dataset["case_ids_by_source"], strict=True)), tmp_path)
    assert artifact_loader._validate_runtime_comparison_binding(payload, frame=frame, role=role) == payload

    for section, key in (("model_identity", "best_checkpoint_sha256"), ("dataset_identity", "dataset_provenance_sha256")):
        mismatched = copy.deepcopy(payload)
        mismatched[section][key] = "mismatch"
        with pytest.raises(ValueError, match="Dataset or checkpoint identity"):
            artifact_loader._validate_runtime_comparison_binding(mismatched, frame=frame, role=role)


def _saved_role(split: Any, name: str, case_ids_by_source: Mapping[str, Sequence[str]], root: Path) -> Any:
    """Return one saved artifact role over the given sources, each holding exactly its listed cases."""
    plan = analysis.artifacts.plan.ArtifactRolePlan(
        split=split,
        dataset_role="id" if split == "eval" else "material_ood",
        dataset_name=name,
        source_dataset_ids=tuple(case_ids_by_source),
        source_identities=tuple({"dataset_id": source, "case_ids": list(ids)} for source, ids in case_ids_by_source.items()),
        case_ids_by_source=tuple(tuple(ids) for ids in case_ids_by_source.values()),
        membership_digests=tuple("a" * 64 for _ in case_ids_by_source),
    )
    return artifact_loader._SavedRole(plan=plan, root=root)


def test_steady_payload_membership_binds_file_names_to_saved_dataset_case_ids(tmp_path: Path) -> None:
    """Admit NPZ files named by the saved Dataset case IDs of every source and reject any other name, order or source."""
    sample_ids = ("lentil__case_0003__fe83e052fbf2", "field_pea__case_0001__0123456789ab")
    names = tuple(f"{case_id}.npz" for case_id in sample_ids)
    (tmp_path / "npz").mkdir()
    for name in names:
        (tmp_path / "npz" / name).write_bytes(b"payload")

    def role(ids: tuple[str, ...]) -> Any:
        return _saved_role("material_ood", "ood", {"ood_a": ids[:1], "ood_b": ids[1:]}, tmp_path)

    def membership(table_names: tuple[str, ...], manifest_names: tuple[str, ...], sources: tuple[str, ...] = ("ood_a", "ood_b")) -> dict[str, Any]:
        frame = pd.DataFrame(
            {
                "source_index": [0, 1],
                "split_local_index": [0, 1],
                "case_index": [1, 2],
                "source_dataset_id": list(sources),
                "npz_path": [str(tmp_path / "npz" / name) for name in table_names],
            }
        )
        outputs = {"parquet": {"path": "artifacts.parquet"}, "npz": [{"path": f"npz/{name}"} for name in sorted(manifest_names)]}
        return {"frame": frame, "provenance": {"outputs": outputs}, "effective_indices": (0, 1)}

    artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names, names))
    with pytest.raises(ValueError, match="do not identify exact run-owned case payloads"):
        artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names[::-1], names))
    with pytest.raises(ValueError, match="manifest NPZ membership"):
        artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names, ("case_0001.npz", "case_0002.npz")))
    with pytest.raises(ValueError, match="source_dataset_id"):
        artifact_loader._validate_payload_membership(role=role(sample_ids), **membership(names, names, ("ood_a", "ood_a")))
    with pytest.raises(ValueError, match="package case ID"):
        artifact_loader._validate_payload_membership(role=role(("case-a", "case-b")), **membership(names, names))


class _SharedSelection:
    """Record exact case/channel publications from steady cached viewers."""

    def __init__(self) -> None:
        self.selection = SimpleNamespace(case_id="2", channels=("p",), datasets=None, pooled=False, focus_dataset=None)
        self.capabilities: Any = None

    def bind_capabilities(self, capabilities: Any) -> None:
        self.capabilities = capabilities

    def select_case(self, case_id: str) -> None:
        self.selection.case_id = case_id

    def select_channels(self, channels: tuple[str, ...]) -> None:
        self.selection.channels = channels

    def select_datasets(self, keys: tuple[Any, ...], *, pooled: bool) -> None:
        self.selection.datasets, self.selection.pooled = keys, pooled

    def select_focus_dataset(self, key: Any) -> None:
        self.selection.focus_dataset = key


def test_steady_case_and_channel_controls_share_one_state_owner(
    tmp_path: Path,
) -> None:
    """Synchronize exact case, channel and dataset state without transient-only controls."""
    frame, _x_grid = steady_frame(tmp_path)
    fields = analysis.evaluation.presentation.display_fields(frame)
    labels = tuple(field.label for field in fields)
    shared = _SharedSelection()
    scope = _scope({"Steady fixture": frame}, comparison=False)

    def make_view() -> tuple[Any, Any, Any]:
        view = panel._indexed_view(
            scope,
            plot_func=lambda **_kwargs: None,
            controls={"channel": analysis.ui.controls.channel_dropdown(labels)},
            selection_state=shared,  # type: ignore[arg-type]
            export={"export_state": None, "export_plot_name": "steady_case", "export_title": None},
            single_model=True,
            sync=True,
            channel_fields=fields,
        )
        case_navigator, channel = cast("widgets.Box", view.rows[1]).children
        return view, case_navigator.children[0], channel

    first, first_case, first_channel = make_view()
    assert first_case.value == _CASE_COUNT
    assert first_channel.value == "p"
    assert first.status.text == "generation case 2"  # the navigator names the opened case's generation case ID
    second, second_case, second_channel = make_view()

    first_case.value = 1
    first_channel.value = "|u|"
    assert shared.selection.case_id == "1"
    assert shared.selection.channels == ("velocity_magnitude",)
    second.deactivate()
    second.activate()
    assert second_case.value == 1
    assert second_channel.value == "|u|"
    descriptions = {str(child.description) for row in (*first.rows, *second.rows) for child in _widget_tree(row) if hasattr(child, "description")}
    assert not {"Time:", "Protocol:", "Horizon:"}.intersection(descriptions)


def _tab_layout(view: widgets.Widget) -> tuple[tuple[str, ...], list[list[str]]]:
    """Return the displayed tab titles and each tab's view titles of one lazy panel."""
    lazy = panel._LAZY_PANELS[view]
    return tuple(cast("Any", lazy.tabs).titles), [list(section.titles) for section in lazy.sections]


def test_steady_tabs_follow_the_decided_order_and_numbers_follow_the_position(tmp_path: Path) -> None:
    """Order the steady tabs per mode, move the configuration table and the capacity view with the mode, and number by position."""
    frame, _grid = steady_frame(tmp_path)
    titles = panel.EVALUATION_SECTION_TITLES
    single_keys = (
        "overview",
        "global_error",
        "error_decomposition",
        "physical_consistency",
        "spectral_analysis",
        "parameter_sensitivity",
        "sample_viewer",
        "outlier_analysis",
        "runtime",
    )
    comparison_keys = (*single_keys[:2], "architecture", *single_keys[2:])
    layouts = {}
    for comparison, keys, datasets in (
        (False, single_keys, {"A": frame}),
        (True, comparison_keys, {"A": frame, "B": frame.copy(deep=True)}),
    ):
        tabs, views = _tab_layout(panel._build_panel(contexts=_contexts_of(datasets, comparison=comparison), comparison=comparison, sections="all"))
        assert tabs == tuple(f"{number}. {titles[key]}" for number, key in enumerate(keys, start=1))
        for number, labels in enumerate(views, start=1):
            assert all(label.startswith(f"{number}-{position}. ") for position, label in enumerate(labels, start=1))
        layouts[comparison] = views
    single, pair = layouts[False], layouts[True]
    assert any("Model-family configuration" in label for label in single[0])
    assert not any("Exact trainable parameters" in label for labels in single for label in labels)
    assert not any("Model-family configuration" in label for label in pair[0])
    architecture = pair[2]
    assert "Model-family configuration" in architecture[0]
    assert "Exact trainable parameters" in architecture[1]

    single_context = _contexts_of({"A": frame}, comparison=False)
    sample_tabs, sample_views = _tab_layout(panel._build_panel(contexts=single_context, comparison=False, sections=("sample_viewer",)))
    assert sample_tabs == (f"1. {titles['sample_viewer']}",)
    assert sample_views[0][0].startswith("1-1. ")
    with pytest.raises(ValueError, match="Unknown evaluation sections"):
        panel._build_panel(contexts=single_context, comparison=False, sections=("architecture",))


def _loaded_artifact(frame: pd.DataFrame, root: Path) -> Any:
    """Wrap one frame as the loaded artifact of a role."""
    return artifact_loader.LoadedEvaluationArtifact(
        split_role="eval",
        dataset_name="steady-fixture-dataset",
        root=root,
        frame=frame,
        identity_sha256="e" * 64,
    )


def _role_frame(directory: Path, case_ids: tuple[int, int], *, split_role: str = "eval") -> pd.DataFrame:
    """Return one two-case steady frame of a split role whose exact case identities are the given ones."""
    directory.mkdir()
    frame, _grid = steady_frame(directory)
    frame["case_index"] = list(case_ids)
    frame.attrs["artifact_provenance"]["split_role"] = split_role
    frame.attrs["dataset_role"] = split_role
    return frame


def test_steady_single_run_with_several_roles_admits_one_case_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit one run's roles, whose cases are disjoint, as their ordered union through the workspace path."""
    id_frame = _role_frame(tmp_path / "id", (1, 2))
    ood_frame = _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")
    run_dir = tmp_path / "run"
    loaded = artifact_loader.LoadedRunArtifacts(
        task="steady_flow",
        run_name="steady-fixture",
        run_dir=run_dir,
        id_artifact=_loaded_artifact(id_frame, run_dir / "analysis" / "id"),
        material_ood_artifact=_loaded_artifact(ood_frame, run_dir / "analysis" / "material_ood"),
    )
    prepared = analysis.artifacts.service.PreparedRunArtifacts(
        loaded_run=loaded,
        role_actions={"id": "reused", "material_ood": "reused"},
    )
    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda _path, **_kwargs: prepared)
    state = _SharedSelection()
    workflow = analysis.evaluation.workflow
    result = workflow.prepare_evaluation_workflow(
        (workflow.EvaluationRunSelection(run_dir),),
        (workflow.EvaluationContextSpec("id", "ID", "id"), workflow.EvaluationContextSpec("material_ood", "Material OOD", "material_ood")),
        selection_state=cast("Any", state),
    )
    assert state.capabilities.case_ids == ("1", "2", "3", "4")
    assert tuple(context.key for context in result.contexts) == ("id", "material_ood")
    result.close()


def test_a_workflow_comparison_with_a_run_selected_by_its_path_names_no_model_in_its_titles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two labelled runs end the title as «A vs B»; with one run selected by its path alone the title names none, its legend the alias."""
    frame = _role_frame(tmp_path / "frame", (1, 2))
    runs = {(tmp_path / name).resolve(): artifact for name, artifact in (("first", frame), ("second-alias", _steady_peer(frame)))}
    prepared = {
        run_dir: analysis.artifacts.service.PreparedRunArtifacts(
            loaded_run=artifact_loader.LoadedRunArtifacts(
                task="steady_flow", run_name=run_dir.name, run_dir=run_dir, id_artifact=_loaded_artifact(artifact, run_dir / "analysis" / "id")
            ),
            role_actions={"id": "reused"},
        )
        for run_dir, artifact in runs.items()
    }
    monkeypatch.setattr(analysis.artifacts.service, "load_prepared_run_artifacts", lambda path, **_kwargs: prepared[path])
    plot = analysis.evaluation.plots.error_behavior.plot_global_error_metrics
    figures: list[Any] = []

    def captured(**kwargs: Any) -> Any:
        figures.append(plot(**kwargs))
        return figures[-1]

    monkeypatch.setattr(analysis.evaluation.plots.error_behavior, "plot_global_error_metrics", captured)
    workflow = analysis.evaluation.workflow
    for labels in (("A", "B"), ("A", None)):
        result = workflow.prepare_evaluation_workflow(
            tuple(workflow.EvaluationRunSelection(run_dir, label=label) for run_dir, label in zip(runs, labels, strict=True)),
            (workflow.EvaluationContextSpec("id", "ID", "id"),),
            sections=("global_error",),
        )
        panel._LAZY_PANELS[result.panel].open()
        result.close()
    named, unnamed = (_figure_texts(figure) for figure in figures)  # the first text is the figure title
    assert named[0].endswith("(A vs B)")
    assert not unnamed[0].endswith(")")  # neither «(A vs second-alias)» nor «(A)»
    assert any("second-alias" in text for text in unnamed[1:])  # the legend still names the run by its alias
    plt.close("all")


def test_steady_comparison_intersects_the_cases_within_each_role(tmp_path: Path) -> None:
    """Keep only the cases every run has in a role, role after role, and refuse a role without any."""
    id_frame = _role_frame(tmp_path / "id", (1, 2))
    ood_frame = _role_frame(tmp_path / "ood", (3, 4))
    first, second = tmp_path / "first", tmp_path / "second"

    def bind(artifacts: dict[tuple[Path, str], pd.DataFrame]) -> _SharedSelection:
        state = _SharedSelection()
        wrapped = {key: SimpleNamespace(frame=value) for key, value in artifacts.items()}
        analysis.evaluation.workflow._bind_steady_capabilities(cast("Any", state), cast("Any", wrapped))
        return state

    overlapping = bind(
        {
            (first, "id"): id_frame,
            (first, "material_ood"): ood_frame,
            (second, "id"): id_frame.iloc[1:],
            (second, "material_ood"): ood_frame,
        }
    )
    assert overlapping.capabilities.case_ids == ("2", "3", "4")
    with pytest.raises(ValueError, match="share no exact case identities"):
        bind({(first, "id"): id_frame, (second, "id"): ood_frame})


def test_the_dataset_choice_of_one_view_reaches_the_others_through_the_selection_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deselecting a dataset in one view redraws it and is published; a view shown later takes it silently."""
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets={"ID": _role_frame(tmp_path / "id", (1, 2))}),
        panel.EvaluationContext(
            key="material_ood", label="Material OOD", datasets={"Material OOD": _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")}
        ),
    ]
    scope = panel._SteadyScope(contexts, comparison=False, refused_roles={})
    shared = _SharedSelection()
    drawn: list[tuple[str, ...]] = []
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda _view, _plot, **kwargs: drawn.append(tuple(kwargs["kwargs"]["datasets"])) or True)

    def make_view() -> Any:
        return panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=shared, export={})  # type: ignore[arg-type]

    first, second = make_view(), make_view()
    assert drawn == [("Lentil, ID", "Lentil, Material OOD")] * 2

    ood = next(item for item in _widget_tree(first.widget) if getattr(item, "description", None) == "Lentil, Material OOD")
    ood.value = False
    assert shared.selection.datasets == (("lentil", "id"),)
    assert drawn[-1] == ("Lentil, ID",)

    second.deactivate()
    second.activate()
    assert drawn[-1] == ("Lentil, ID",)
    assert [item.value for item in _widget_tree(second.widget) if getattr(item, "description", None) == "Lentil, Material OOD"] == [False]
    assert shared.selection.datasets == (("lentil", "id"),)


def test_a_case_count_viewer_rebuilt_for_other_datasets_resumes_the_count_it_showed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The count the user chose survives a dataset change, and the replaced viewer stops following the controls it shared."""
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets={"ID": _role_frame(tmp_path / "id", (1, 2))}),
        panel.EvaluationContext(
            key="material_ood", label="Material OOD", datasets={"Material OOD": _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")}
        ),
    ]
    scope = panel._SteadyScope(contexts, comparison=False, refused_roles={})
    built: list[Any] = []
    original = analysis.ui.viewers.case_count_view

    def recording(*args: Any, **kwargs: Any) -> Any:
        built.append((kwargs.get("resume_cases"), original(*args, **kwargs)))
        return built[-1][1]

    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", recording)
    view = panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=None, export={}, count=(50, 50), controls=None)
    ((first_resume, first),) = built
    assert first_resume is None
    first.rows[0].children[0].children[0].value = 1
    ood = next(item for item in _widget_tree(view.widget) if getattr(item, "description", None) == "Lentil, Material OOD")
    ood.value = False

    assert [resume for resume, _viewer in built] == [None, 1]
    assert not first.active
    assert built[1][1].active


@pytest.mark.parametrize("count", [None, (50, 50)], ids=["plain", "nested-count-viewer"])
def test_a_view_hidden_from_inside_its_own_plot_publishes_nothing_and_draws_again_when_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: tuple[int, int] | None
) -> None:
    """A draw superseded by hiding the view leaves the plot untouched; showing the view draws the missing one, nested viewer included."""
    contexts = [
        panel.EvaluationContext(key="id", label="ID", datasets={"ID": _role_frame(tmp_path / "id", (1, 2))}),
        panel.EvaluationContext(
            key="material_ood", label="Material OOD", datasets={"Material OOD": _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")}
        ),
    ]
    scope = panel._SteadyScope(contexts, comparison=False, refused_roles={})
    hide: list[Any] = []

    def plot(**_kwargs: object) -> str:
        for view in hide:
            view.deactivate()
        hide.clear()
        return "drawn"

    shown = _record_shows(monkeypatch)
    view = panel._dataset_view(scope, plot_func=plot, selection_state=None, export={}, count=count)
    assert "drawn" in _taken(shown)
    ood = next(item for item in _widget_tree(view.widget) if getattr(item, "description", None) == "Lentil, Material OOD")
    hide.append(view)
    ood.value = False
    assert "drawn" not in _taken(shown)
    assert not view.active
    view.activate()
    assert "drawn" in _taken(shown)


def _two_role_scope(tmp_path: Path) -> Any:
    """Return the single-model scope of an ID and a material-OOD frame: two datasets (Lentil, ID and Lentil, Material OOD)."""
    return panel._SteadyScope(
        [
            panel.EvaluationContext(key="id", label="ID", datasets={"ID": _role_frame(tmp_path / "id", (1, 2))}),
            panel.EvaluationContext(
                key="material_ood", label="Material OOD", datasets={"Material OOD": _role_frame(tmp_path / "ood", (3, 4), split_role="material_ood")}
            ),
        ],
        comparison=False,
        refused_roles={},
    )


def test_an_explicitly_empty_dataset_selection_is_shared_as_a_choice_not_as_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing every dataset in one view clears them in a view shown later, silently, instead of leaving its old selection."""
    scope = _two_role_scope(tmp_path)
    shared = _SharedSelection()
    monkeypatch.setattr(analysis.ui.viewers, "render_view", lambda *_args, **_kwargs: True)

    def make_view() -> Any:
        return panel._dataset_view(scope, plot_func=lambda **_kwargs: None, selection_state=shared, export={})  # type: ignore[arg-type]

    def boxes(view: Any) -> list[Any]:
        return [item for item in _widget_tree(view.widget) if getattr(item, "description", None) in {"Lentil, ID", "Lentil, Material OOD"}]

    first, second = make_view(), make_view()
    assert shared.selection.datasets is None
    for box in boxes(first):
        box.value = False
    assert shared.selection.datasets == ()

    second.deactivate()
    second.activate()
    assert [box.value for box in boxes(second)] == [False, False]
    assert shared.selection.datasets == ()


def test_a_counted_view_hidden_and_shown_again_still_follows_its_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hiding the view supersedes its render; the viewer nested in it is bound to the render shown again and draws what changed meanwhile."""
    scope = _two_role_scope(tmp_path)
    built: list[Any] = []
    original = analysis.ui.viewers.case_count_view

    def recording(*args: Any, **kwargs: Any) -> Any:
        built.append(original(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(analysis.ui.viewers, "case_count_view", recording)
    shown = _record_shows(monkeypatch)
    view = panel._dataset_view(scope, plot_func=lambda **_kwargs: "drawn", selection_state=None, export={}, count=(50, 50))
    (nested,) = built
    _taken(shown)

    view.deactivate()
    nested.rows[0].children[0].children[0].value = 1  # a count change made while hidden is held back
    assert "drawn" not in _taken(shown)
    view.activate()
    assert "drawn" in _taken(shown)
    nested.rows[0].children[0].children[0].value = 2
    assert "drawn" in _taken(shown)
    assert len(built) == 1


def test_the_outlier_view_offers_its_reference_and_names_the_export_by_the_selection_it_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With two cases the plot still offers the two worst and the reference; the file name says which one, not a frame row."""
    frame, _grid = steady_frame(tmp_path)
    monkeypatch.setattr(analysis.evaluation.plots.samples_outliers, "plot_linked_outlier_cases", lambda **_kwargs: None)
    registry = panel._build_sections(_scope({"Steady fixture": frame}, comparison=False), comparison=False)
    ((title, factory, _stem),) = [
        entry for entries, _name in registry.values() for entry in entries if entry[0] == "Worst per-channel cases (field plots)"
    ]
    state: dict[str, Any] = {"fig": None}

    view = analysis.ui.notebook._invoke_dropdown_entry(factory, export_state=state, export_plot_name=title, export_title=title)

    case_field = view.rows[1].children[0].children[0]
    assert "worst_1" in state["filename_stem"]
    case_field.value = _CASE_COUNT + 1
    assert case_field.value == _CASE_COUNT + 1
    assert state["filename_stem"].split("__")[3] == "reference"
    case_field.value = _CASE_COUNT
    assert state["filename_stem"].split("__")[3] == f"worst_{_CASE_COUNT}"


def test_a_steady_role_without_a_shared_case_is_refused_alone(tmp_path: Path) -> None:
    """Both runs keep the ID role while their disjoint material-OOD cases refuse that role with its reason."""
    id_frame = _role_frame(tmp_path / "id", (1, 2))
    ood_frames = (
        _role_frame(tmp_path / "ood_first", (3, 4), split_role="material_ood"),
        _role_frame(tmp_path / "ood_second", (5, 6), split_role="material_ood"),
    )

    def loaded(ood_frame: pd.DataFrame, run_dir: Path) -> Any:
        artifacts = {"id": SimpleNamespace(frame=id_frame), "material_ood": SimpleNamespace(frame=ood_frame)}
        return SimpleNamespace(artifact=artifacts.get, run_dir=run_dir)

    runs = (loaded(ood_frames[0], tmp_path / "first"), loaded(ood_frames[1], tmp_path / "second"))
    roles, refused = analysis.evaluation.workflow._admit_roles(("id", "material_ood", "parameter_ood"), runs, task="steady_flow")

    assert roles == ("id",)
    assert set(refused) == {"material_ood"}
    assert "share no exact case identities" in refused["material_ood"]
