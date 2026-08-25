# ruff: noqa: ANN001, PLR2004, RUF001, S101, SLF001
"""Protect contextual notebook PDF export naming."""

from __future__ import annotations

import ipywidgets as widgets
import matplotlib.pyplot as plt

from src.analysis.ui import analysis_ui_components as components
from src.analysis.ui import analysis_ui_notebook as notebook_ui


def _export_button(panel: widgets.VBox) -> widgets.Button:
    """Return the rendered panel's PDF export trigger."""
    return next(child for child in panel.children[0].children if isinstance(child, widgets.Button) and child.description == "Export PDF")


def test_export_prefers_sanitized_viewer_stem_and_never_overwrites(tmp_path, monkeypatch) -> None:
    """Use viewer context over generic names while retaining timestamp collision protection."""
    displayed: list[object] = []
    monkeypatch.setattr(notebook_ui, "display", displayed.append)
    monkeypatch.setattr(notebook_ui, "clear_output", lambda **_kwargs: None)
    figure = plt.figure()
    state = {
        "fig": figure,
        "figures": (),
        "plot_name": "Generic Plot",
        "title": "plot",
        "filename_stem": "Drying / ID: Case #7",
        "filename_prefix": "ignored-prefix",
    }
    notebook_ui.make_lazy_panel_with_tabs((widgets.VBox(),), export_state=state, export_dir=str(tmp_path))
    opener = displayed[-1]
    assert isinstance(opener, widgets.Button)
    opener.click()
    rendered = displayed[-1]
    assert isinstance(rendered, widgets.VBox)
    button = _export_button(rendered)
    button.click()
    button.click()
    names = sorted(path.name for path in tmp_path.glob("*.pdf"))
    assert len(names) == 2
    assert names[0].startswith("drying_id_case_7_")
    assert names[0] != names[1]
    plt.close(figure)


def test_export_stem_uses_prefix_then_generic_fallback() -> None:
    """Keep legacy prefix and generic plot names backward-compatible."""
    assert notebook_ui._export_stem({"filename_prefix": "Transient – OOD", "plot_name": "generic"}) == "transient_ood_generic"
    assert notebook_ui._export_stem({"plot_name": "Generic / Plot"}) == "generic_plot"


def test_compact_view_case_channel_and_scope_controls_match_eda_contract() -> None:
    """Protect the shared widget classes, dimensions, ordering, and rebind state."""
    section = notebook_ui.make_dropdown_section(
        [("First", lambda: None, "first")],
        select_first=True,
    )
    view = section.children[0]
    assert isinstance(view, widgets.Dropdown)
    assert view.description == ""
    assert view.layout.width == "230px"

    case, previous, following = components.ui_step_case_index(n_cases=3)
    row = components.ui_compact_case_row(case, previous, following)
    assert isinstance(case, widgets.BoundedIntText)
    assert row.layout.width == "230px"
    assert row.layout.flex_flow == "row nowrap"
    assert row.layout.grid_gap == "0"
    assert case.layout.width == "150px"
    assert previous.layout.width == following.layout.width == "40px"
    assert previous.tooltip == "Previous shared case"
    assert following.tooltip == "Next shared case"

    retained: dict[str, tuple[str, ...] | None] = {"value": None}
    state = components.ChannelCheckboxState(
        title="Channels",
        callback=lambda: None,
        selection_getter=lambda: retained["value"],
        selection_setter=lambda values: retained.__setitem__("value", values),
    )
    state.rebind(("T", "phi", "w_surf", "w_int"))
    group = state.group
    assert group is not None
    assert group.layout.width == "max-content"
    grid = group.children[0]
    assert isinstance(grid, widgets.GridBox)
    assert grid.layout.grid_template_columns.startswith("repeat(3,")
    assert tuple(group.boxes) == ("T", "phi", "w_surf", "w_int")
    group.boxes["phi"].value = False
    state.rebind(("T", "w_surf", "future"), labels={"future": "Future [1]"})
    assert state.selected == ("T", "w_surf", "future")
    assert "phi" not in state.status.value
    group = state.group
    assert group is not None
    group.boxes["w_surf"].value = False
    state.rebind(("T",))
    assert state.selected == ("T",)
    assert "future" in state.status.value

    scope = components.ui_scope_toggle()
    scope_row, detail = components.ui_compact_scope_controls(scope)
    assert tuple(scope.options) == (("Aggregate", "aggregate"), ("Single case", "single"))
    aggregate = widgets.IntSlider()
    components.ui_set_scope_detail(detail, (aggregate,))
    assert detail.children == (aggregate,)
    components.ui_set_scope_detail(detail, tuple(row.children))
    assert detail.children == row.children
    assert scope_row.children == (scope, detail)
